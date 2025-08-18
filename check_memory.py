import torch
import types
import argparse
from wan.video2video import CustomWanVace
from wan.modules.model import WanAttentionBlock
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
import os

# --- 로깅 유틸리티 ---
def format_mem(size_bytes):
    """메모리 크기를 GB 단위로 변환"""
    return f"{size_bytes / (1024**3):.2f} GB"

def log_memory(device, stage=""):
    """현재 및 최대 메모리 사용량 로깅"""
    print(f"--- {stage} ---")
    current_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    print(f"Current VRAM: {format_mem(current_mem)}")
    print(f"Peak VRAM:    {format_mem(peak_mem)}")
    print("--------------------")

# --- 몽키 패칭을 위한 새로운 forward 함수 ---
def patched_attention_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    device = x.device
    torch.cuda.synchronize(device)
    
    print("\n[MEMORY LOG] >>> Entering WanAttentionBlock.self_attn (RopE inside)")
    torch.cuda.reset_peak_memory_stats(device)
    before_mem = torch.cuda.memory_allocated(device)
    log_memory(device, "Before self_attn")

    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        seq_lens, grid_sizes, freqs)
    
    torch.cuda.synchronize(device)
    
    after_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    print("\n[MEMORY LOG] <<< Exiting WanAttentionBlock.self_attn")
    log_memory(device, "After self_attn")
    print(f"동적 할당량 (Peak - Before): {format_mem(peak_mem - before_mem)}")
    print("--------------------\n")

    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

    def cross_attn_ffn(x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x

    x = cross_attn_ffn(x, context, context_lens, e)
    return x

def _parse_args():
    parser = argparse.ArgumentParser(description="Memory check for Wan V2V generation")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Wan2.2 base model directory.")
    parser.add_argument("--vace-ckpt-path", type=str, required=True, help="Wan2.1-VACE checkpoint file or directory.")
    parser.add_argument("--task", type=str, default="ti2v-5B", choices=['t2v-A14B', 'i2v-A14B', 'ti2v-5B'], help="Task type.")
    parser.add_argument("--size", type=str, default="1280*720", help="Resolution of the video.")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    device = torch.device("cuda:0")
    config = WAN_CONFIGS[args.task]

    print("Initializing model on CPU...")
    wan_vace = CustomWanVace(
        config=config,
        wan2_2_ckpt_dir=args.ckpt_dir,
        wan2_1_vace_ckpt_path=args.vace_ckpt_path,
        device_id=0,
        init_on_cpu=True,
        offload_model=False
    )
    print("Model initialized.")

    print("\n--- Measuring Static Memory ---")
    print("Moving low_noise_model to GPU...")
    wan_vace.low_noise_model.to(device)
    log_memory(device, "low_noise_model Loaded")
    
    wan_vace.low_noise_model.cpu()
    torch.cuda.empty_cache()
    print("Moved model back to CPU for patching.")

    print("\n--- Applying Monkey Patch for Dynamic Memory Logging ---")
    for i, block in enumerate(wan_vace.low_noise_model.blocks):
        block.original_forward = block.forward
        block.forward = types.MethodType(patched_attention_forward, block)
    print(f"Patched {len(wan_vace.low_noise_model.blocks)} blocks in low_noise_model.")
    
    for i, block in enumerate(wan_vace.high_noise_model.blocks):
        block.original_forward = block.forward
        block.forward = types.MethodType(patched_attention_forward, block)
    print(f"Patched {len(wan_vace.high_noise_model.blocks)} blocks in high_noise_model.")

    print("\n--- Running generation to trigger patched functions ---")
    
    width, height = map(int, args.size.split('*'))
    dummy_frames = [torch.randn(3, 81, height, width, device=device)]
    dummy_masks = [torch.ones(1, 81, height, width, device=device)]
    dummy_ref_images = [None]
    
    try:
        wan_vace.generate(
            input_prompt="a beautiful girl",
            input_frames=dummy_frames,
            input_masks=dummy_masks,
            input_ref_images=dummy_ref_images,
            sampling_steps=10,
            offload_model=False
        )
    except RuntimeError as e:
        print(f"\n--- CAUGHT EXPECTED ERROR ---")
        print(f"Error message: {e}")
        print("This is expected. Check the memory logs above.")
        log_memory(device, "At time of error")

if __name__ == "__main__":
    main()
