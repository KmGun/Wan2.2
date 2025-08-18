
import argparse
import logging
import os
import sys
import torch
import torch.profiler

# Add project root to path to allow imports
sys.path.insert(0, os.getcwd())

from wan.video2video import CustomWanVace
from wan.configs.wan_ti2v_5B import ti2v_5B as ti2v_5b_config
from wan.configs.wan_i2v_A14B import i2v_A14B as i2v_a14b_config
from wan.configs.wan_t2v_A14B import t2v_A14B as t2v_a14b_config

# --- Utility Functions ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, required=True, help="Path to the main checkpoint directory for Wan2.2.")
    parser.add_argument('--vace-ckpt-path', type=str, required=True, help="Path to the VACE checkpoint file (e.g., diffusion_pytorch_model.safetensors).")
    parser.add_argument('--vae-ckpt-path', type=str, required=True, help="Path to the VAE checkpoint file (e.g., Wan2.1_VAE.pth).")
    parser.add_argument('--input-video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output-path', type=str, default='./result.mp4', help="Path to save the generated video.")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument('--n-prompt', type=str, default="", help="Negative prompt.")
    parser.add_argument('--task', type=str, default="ti2v-5B", choices=["ti2v-5B", "i2v-A14B", "t2v-A14B"], help="Task type.")
    parser.add_argument('--size', type=str, default="704*1280", help="Resolution of the output video (e.g., '704*1280').")
    parser.add_argument('--num-frames', type=int, default=16, help="Number of frames to generate.")
    parser.add_argument('--fps', type=int, default=8, help="FPS of the output video.")
    parser.add_argument('--seed', type=int, default=-1, help="Random seed.")
    parser.add_argument('--offload-model', action='store_true', help="Offload model to CPU to save VRAM.")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()

    logging.info(f"인자 파싱 완료. 태스크: {args.task}, 프롬프트: {args.prompt}")

    # --- Configuration ---
    if args.task == 'ti2v-5B':
        config = ti2v_5b_config
    elif args.task == 'i2v-A14B':
        config = i2v_a14b_config
    else:
        config = t2v_a14b_config
    
    config.param_dtype = torch.float16 # Ensure float16 for memory efficiency
    logging.info(f"설정 불러오기 완료. 샘플 FPS: {config.sample_fps}")

    # --- Model Initialization ---
    logging.info("CustomWanVace 모델 초기화 시작...")
    wan_vace = CustomWanVace(
        config,
        wan2_2_ckpt_dir=args.ckpt_dir,
        wan2_1_vace_ckpt_path=args.vace_ckpt_path,
        vae_ckpt_path=args.vae_ckpt_path,
        offload_model=args.offload_model
    )
    logging.info("모델 초기화 완료")

    # --- Data Preparation ---
    logging.info("소스 데이터 준비 시작...")
    image_size = [int(s) for s in args.size.split('*')]
    input_frames, input_masks, input_ref_images = wan_vace.prepare_source(
        [args.input_video], [None], [None],
        num_frames=args.num_frames,
        image_size=image_size,
        device=wan_vace.device
    )
    logging.info(f"입력 비디오: {args.input_video}, 프레임 수: {len(input_frames[0]) if input_frames else 0}, 해상도: {args.size}")
    logging.info("소스 데이터 준비 완료")

    # --- Profiling Video Generation ---
    logging.info("프로파일러로 비디오 생성 시작...")
    
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                 wan_vace.generate(
                    input_frames=input_frames,
                    input_masks=input_masks,
                    input_ref_images=input_ref_images,
                    input_prompt=args.prompt,
                    n_prompt=args.n_prompt,
                    sampling_steps=50, # Use same steps as original command
                    offload_model=args.offload_model,
                    seed=args.seed,
                )

    except torch.cuda.OutOfMemoryError as e:
        logging.error("CUDA Out of Memory 에러 발생! 프로파일러 결과를 출력합니다.")
        logging.error(e)
    
    finally:
        if 'prof' in locals():
            logging.info("--- CUDA Memory Usage by Operator (Top 15) ---")
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=15))
            
            logging.info("--- CUDA Memory Usage by Operator (Grouped by Stack) ---")
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=15))


if __name__ == '__main__':
    main()
