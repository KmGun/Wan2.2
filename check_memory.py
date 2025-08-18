
import torch
import logging
import sys
import os
# Add project root to path to allow imports
sys.path.insert(0, os.getcwd())

from wan.configs.wan_ti2v_5B import config
from wan.modules.vace_v2v_model import CustomVaceWanModel
from wan.modules.vae import WanVAE
from wan.text2video import T5EncoderModel

def print_memory_usage(stage_name):
    if not torch.cuda.is_available():
        print(f"[MEMORY_CHECK] Stage: {stage_name} - CUDA not available.")
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"---[MEMORY_CHECK] Stage: {stage_name} ---")
    print(f"  - Allocated: {allocated:.2f} MB")
    print(f"  - Reserved:  {reserved:.2f} MB")
    print("-" * 40)

def check_memory():
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting memory check.")
        return

    device = torch.device("cuda:0")
    ckpt_dir = './checkpoints/Wan2.2-TI2V-5B'
    vace_ckpt_path = './checkpoints/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors'
    vae_ckpt_path = './checkpoints/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth'
    param_dtype = config.param_dtype

    try:
        print_memory_usage("Initial state")

        # 1. Load T5 Encoder
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=device, # Load directly to GPU to measure
            checkpoint_path=os.path.join(ckpt_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(ckpt_dir, config.t5_tokenizer),
        )
        print_memory_usage("After loading T5 Encoder")
        del text_encoder
        torch.cuda.empty_cache()
        print_memory_usage("After deleting T5 Encoder")


        # 2. Load VAE
        vae = WanVAE(vae_pth=vae_ckpt_path, device=device)
        print_memory_usage("After loading VAE")
        del vae
        torch.cuda.empty_cache()
        print_memory_usage("After deleting VAE")

        # 3. Load Low Noise Model
        low_noise_model = CustomVaceWanModel.from_pretrained(
            ckpt_dir, subfolder=config.low_noise_checkpoint
        ).to(device).to(param_dtype)
        print_memory_usage("After loading Low Noise Model")
        del low_noise_model
        torch.cuda.empty_cache()
        print_memory_usage("After deleting Low Noise Model")

        # 4. Load High Noise Model
        high_noise_model = CustomVaceWanModel.from_pretrained(
            ckpt_dir, subfolder=config.high_noise_checkpoint
        ).to(device).to(param_dtype)
        print_memory_usage("After loading High Noise Model")
        del high_noise_model
        torch.cuda.empty_cache()
        print_memory_usage("After deleting High Noise Model")
        
        # 5. Load all main components together
        print("\n--- Loading all components together ---")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=device,
            checkpoint_path=os.path.join(ckpt_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(ckpt_dir, config.t5_tokenizer),
        )
        vae = WanVAE(vae_pth=vae_ckpt_path, device=device)
        low_noise_model = CustomVaceWanModel.from_pretrained(
            ckpt_dir, subfolder=config.low_noise_checkpoint
        ).to(device).to(param_dtype)
        high_noise_model = CustomVaceWanModel.from_pretrained(
            ckpt_dir, subfolder=config.high_noise_checkpoint
        ).to(device).to(param_dtype)
        print_memory_usage("After loading ALL main models")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_memory()
