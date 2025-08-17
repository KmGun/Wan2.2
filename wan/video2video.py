import gc
import logging
import math
import os
import random
import sys
import time
import traceback
import types
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


from .modules.vace_v2v_model import CustomVaceWanModel
from .modules.vae import WanVAE
from .text2video import (
    FlowDPMSolverMultistepScheduler,
    FlowUniPCMultistepScheduler,
    T5EncoderModel,
    WanT2V,
    get_sampling_sigmas,
    retrieve_timesteps,
    shard_model,
)
from contextlib import contextmanager
from .text2video import WanT2V

class CustomWanVace(WanT2V):
    def __init__(
        self,
        config,
        wan2_2_ckpt_dir: str,
        wan2_1_vace_ckpt_path : str,
        vae_ckpt_path: str = None,
        device_id = 0,
        **kwargs
    ):
        # GPU
        self.device = torch.device(f"cuda:{device_id}")

        # 기본 파라미터
        self.config = config
        self.rank = kwargs.get('rank', 0)
        self.param_dtype = config.param_dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.sp_size = 1
        self.patch_size = config.patch_size
        self.vae_stride = config.vae_stride  # Add vae_stride from config
        self.sample_neg_prompt = config.sample_neg_prompt
        self.init_on_cpu = kwargs.get('init_on_cpu', True)
        self.offload_model = kwargs.get('offload_model', True)
        self.vae_stride = config.vae_stride

        # T5 인코더 준비
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(wan2_2_ckpt_dir,config.t5_checkpoint),
            tokenizer_path=os.path.join(wan2_2_ckpt_dir,config.t5_tokenizer),
        )

        # VAE 준비
        vae_path = vae_ckpt_path if vae_ckpt_path else os.path.join(wan2_2_ckpt_dir, config.vae_checkpoint)
        self.vae = WanVAE(
            vae_pth=vae_path,
            device=self.device
        )

        # 2개 MoE DiT 모델 준비
        logging.info(f"Creating CustomVaceWanModel from {wan2_2_ckpt_dir}")
        self.low_noise_model = CustomVaceWanModel.from_pretrained(
            wan2_2_ckpt_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_vace_model(
            model=self.low_noise_model,
            vace_ckpt_path=wan2_1_vace_ckpt_path,
            keep_on_cpu=self.init_on_cpu)
        
        self.high_noise_model = CustomVaceWanModel.from_pretrained(
            wan2_2_ckpt_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_vace_model(
            model=self.high_noise_model,
            vace_ckpt_path=wan2_1_vace_ckpt_path,
            keep_on_cpu=self.init_on_cpu)

    def generate (self,
                  input_frames,
                  input_ref_images,
                  input_masks,
                  input_prompt,
                  n_prompt="",
                  sample_solver='unipc',
                  sampling_steps=50,
                  offload_model=True,
                  shift=5.0,
                  context_scale=1.0,
                  seed=-1
            
    ):
        ## 1단계 준비

        # n_prompt
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # seed_g : 시드 생성
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # context : 프롬프트 임베딩
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()

        # z, m0, z0 : input 비디오, 마스크, 레퍼런스 이미지 처리
        z0 = self.vace_encode_frames(
            input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)

        # noise : seed_g 기반으로 노이즈 생성.
        target_shape = list(z0[0].shape)
        if input_masks is not None:
            target_shape[0] = int(target_shape[0] / 2)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        # 영상 총 길이 계산
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        ## 2단계 - 기타 설정
        # 멀티 GPU 연산 설정
        @contextmanager
        def noop_no_sync():
            yield
        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',noop_no_sync)
        # 스케줄러 설정 (노이즈 제거 방법)
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync_low_noise(), no_sync_high_noise:

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            ## 3단계 - 비디오 생성

            # 기본 변수
            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # DiT 루프
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                # timestep에 따라 모델 스위치.
                model = super()._prepare_model_for_timestep(t, self.boundary * self.num_train_timesteps, offload_model)

                noise_pred_cond = model(
                    latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    **arg_c)[0]
                noise_pred_uncond = model(
                    latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    **arg_null)[0]

                # CFG 처리
                current_guide_scale = self.config.sample_guide_scale[1] if t.item() >= self.boundary * self.num_train_timesteps else self.config.sample_guide_scale[0]
                noise_pred = noise_pred_uncond + current_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # 디노이즈 : step()
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            # 마무리
            x0 = latents
            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        # 4단계 - 비디오 생성 완료
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [
                torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)
            ]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [
                        torch.cat((u, torch.zeros_like(u)), dim=0)
                        for u in ref_latent
                    ]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        # HACK: The vae_stride value from the config seems incorrect for this model (ti2v-5B).
        # The VAE uses a stride of 8 (e.g., 1280 -> 160), but the stride for width
        # appears to be 16, causing a mismatch. Overriding to the correct value.
        vae_stride = [1, 8, 8]
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = int(height) // vae_stride[1]
            width = int(width) // vae_stride[2]

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(depth, height, vae_stride[1], width,
                             vae_stride[1])  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height,
                                width)  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(new_depth, height, width),
                mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]
    
    def decode_latent(self, zs, ref_images=None, vae=None):

        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return vae.decode(trimed_zs)
    
    def _configure_vace_model(self, model, vace_ckpt_path, keep_on_cpu=False):
        """
        Configures a VACE model with additional VACE weights.
        
        Args:
            model (torch.nn.Module):
                The model instance to configure.
            vace_ckpt_path (str):
                Path to VACE checkpoint file.
        
        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        
        # Wan 2.1 VACE 웨이트만 추출하여 덮어쓰기
        # Handle both .bin and .safetensors formats
        if vace_ckpt_path.endswith('.bin'):
            vace_full_state_dict = torch.load(vace_ckpt_path, map_location='cpu')
        elif vace_ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            vace_full_state_dict = load_file(vace_ckpt_path)
        else:
            # If path is a directory, try to find the checkpoint file
            import os
            if os.path.isdir(vace_ckpt_path):
                safetensors_path = os.path.join(vace_ckpt_path, 'diffusion_pytorch_model.safetensors')
                if os.path.exists(safetensors_path):
                    from safetensors.torch import load_file
                    vace_full_state_dict = load_file(safetensors_path)
                else:
                    raise FileNotFoundError(f"No checkpoint file found in {vace_ckpt_path}")
            else:
                raise ValueError(f"Unknown checkpoint format: {vace_ckpt_path}")
        vace_filtered_state_dict = {
            k: v for k, v in vace_full_state_dict.items()
            if k.startswith('vace_blocks.') or k.startswith('vace_patch_embedding.')
        }
        
        if vace_filtered_state_dict:
            model.load_state_dict(vace_filtered_state_dict, strict=False)
            logging.info(f"Loaded VACE weights from {vace_ckpt_path}")
        else:
            logging.warning("No VACE weights found in checkpoint")
        
        if not keep_on_cpu:
            model.to(self.device)
        return model

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames,
                       image_size, device):
        import imageio
        import numpy as np
        
        # Validate image size
        area = image_size[0] * image_size[1]
        if area not in [720 * 1280, 480 * 832, 704 * 1280]:
            raise NotImplementedError(
                f'image_size {image_size} is not supported')

        # Swap dimensions for processing (height, width)
        image_size = (image_size[1], image_size[0])
        image_sizes = []
        
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                # Load video and mask pair
                video_reader = imageio.get_reader(sub_src_video)
                mask_reader = imageio.get_reader(sub_src_mask)
                
                video_frames = []
                mask_frames = []
                
                for frame_idx in range(min(num_frames, len(video_reader))):
                    # Load video frame
                    frame = video_reader.get_data(frame_idx)
                    frame = Image.fromarray(frame).convert("RGB")
                    frame = frame.resize(image_size[::-1], Image.LANCZOS)
                    frame = TF.to_tensor(frame).sub_(0.5).div_(0.5)
                    video_frames.append(frame)
                    
                    # Load mask frame
                    mask = mask_reader.get_data(frame_idx)
                    if len(mask.shape) == 3:
                        mask = mask[:,:,0]  # Take first channel if RGB
                    mask = Image.fromarray(mask).convert("L")
                    mask = mask.resize(image_size[::-1], Image.LANCZOS)
                    mask = TF.to_tensor(mask)
                    mask_frames.append(mask)
                
                video_reader.close()
                mask_reader.close()
                
                # Pad if needed
                while len(video_frames) < num_frames:
                    video_frames.append(video_frames[-1])
                    mask_frames.append(mask_frames[-1])
                
                src_video[i] = torch.stack(video_frames, dim=1).to(device)
                src_mask[i] = torch.stack(mask_frames, dim=1).to(device)
                src_mask[i] = torch.clamp(src_mask[i], min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
                
            elif sub_src_video is None:
                # Create empty video with mask
                src_video[i] = torch.zeros(
                    (3, num_frames, image_size[0], image_size[1]),
                    device=device)
                src_mask[i] = torch.ones_like(src_video[i][:1], device=device)
                image_sizes.append(image_size)
                
            else:
                # Load video only (no mask)
                video_reader = imageio.get_reader(sub_src_video)
                video_frames = []
                
                for frame_idx in range(min(num_frames, len(video_reader))):
                    frame = video_reader.get_data(frame_idx)
                    frame = Image.fromarray(frame).convert("RGB")
                    frame = frame.resize(image_size[::-1], Image.LANCZOS)
                    frame = TF.to_tensor(frame).sub_(0.5).div_(0.5)
                    video_frames.append(frame)
                
                video_reader.close()
                
                # Pad if needed
                while len(video_frames) < num_frames:
                    video_frames.append(video_frames[-1])
                
                src_video[i] = torch.stack(video_frames, dim=1).to(device)
                src_mask[i] = torch.ones((1, num_frames, image_size[0], image_size[1]), 
                                        device=device)
                image_sizes.append(src_video[i].shape[2:])

        # Process reference images
        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(
                            0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones(
                                (3, 1, canvas_height, canvas_width),
                                device=device)  # [-1, 1]
                            scale = min(canvas_height / ref_height,
                                        canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(
                                ref_img.squeeze(1).unsqueeze(0),
                                size=(new_height, new_width),
                                mode='bilinear',
                                align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height,
                                         left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images
