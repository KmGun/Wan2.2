import torch
import types
from wan.video2video import CustomWanVace
from wan.modules.model import WanAttentionBlock
from wan.configs import WAN_CONFIGS
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
# 원래의 WanAttentionBlock.forward 함수를 감싸서 메모리 로깅을 추가합니다.
def patched_attention_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    device = x.device
    torch.cuda.synchronize(device)
    
    # RopE 연산이 포함된 self-attention 직전 메모리 측정
    print("\n[MEMORY LOG] >>> Entering WanAttentionBlock.self_attn (RopE inside)")
    torch.cuda.reset_peak_memory_stats(device)
    before_mem = torch.cuda.memory_allocated(device)
    log_memory(device, "Before self_attn")

    # 원래의 self-attention 연산 수행
    # self.self_attn 내부의 rope_apply에서 메모리 사용량이 급증할 것으로 예상됩니다.
    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        seq_lens, grid_sizes, freqs)
    
    torch.cuda.synchronize(device)
    
    # self-attention 직후 메모리 측정
    after_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    print("\n[MEMORY LOG] <<< Exiting WanAttentionBlock.self_attn")
    log_memory(device, "After self_attn")
    print(f"동적 할당량 (Peak - Before): {format_mem(peak_mem - before_mem)}")
    print("--------------------")

    # 원래 함수의 나머지 부분 실행
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


def main():
    # --- 설정 (generate_v2v.py와 유사하게 설정) ---
    # 사용자의 환경에 맞게 경로를 수정해야 합니다.
    ckpt_dir = "./checkpoints/Wan2.2-TI2V-5B" 
    vace_ckpt_path = "./checkpoints/Wan2.1-VACE-1.3B/"
    
    if not os.path.exists(ckpt_dir) or not os.path.exists(vace_ckpt_path):
        print("ERROR: Model checkpoint paths are not correct. Please edit them in check_memory.py")
        return

    device = torch.device("cuda:0")
    config = WAN_CONFIGS['ti2v-5B']

    # --- 모델 초기화 (CPU) ---
    print("Initializing model on CPU...")
    wan_vace = CustomWanVace(
        config=config,
        wan2_2_ckpt_dir=ckpt_dir,
        wan2_1_vace_ckpt_path=vace_ckpt_path,
        device_id=0,
        init_on_cpu=True,
        offload_model=False # Offload를 꺼야 모델이 GPU에 계속 상주하여 측정 가능
    )
    print("Model initialized.")

    # --- 1. 정적 메모리 측정 ---
    print("\n--- Measuring Static Memory ---")
    print("Moving low_noise_model to GPU...")
    wan_vace.low_noise_model.to(device)
    log_memory(device, "low_noise_model Loaded")
    
    # 측정이 끝났으면 다시 CPU로 보내서 generate 준비
    wan_vace.low_noise_model.cpu()
    torch.cuda.empty_cache()
    print("Moved model back to CPU for patching.")


    # --- 2. 동적 메모리 측정을 위한 몽키 패칭 ---
    print("\n--- Applying Monkey Patch for Dynamic Memory Logging ---")
    # low_noise_model의 모든 WanAttentionBlock의 forward 함수를 우리가 만든 함수로 교체
    for i, block in enumerate(wan_vace.low_noise_model.blocks):
        block.original_forward = block.forward # 만약을 위해 원본 저장
        block.forward = types.MethodType(patched_attention_forward, block)
        print(f"Patched block {i}")
    
    # high_noise_model도 동일하게 패치
    for i, block in enumerate(wan_vace.high_noise_model.blocks):
        block.original_forward = block.forward
        block.forward = types.MethodType(patched_attention_forward, block)


    # --- 3. 생성 실행 및 동적 메모리 측정 ---
    print("\n--- Running generation to trigger patched functions ---")
    # generate_v2v.py와 유사한 더미 입력 생성
    # 실제 비디오/이미지 경로를 사용하거나, 더미 텐서를 직접 만듭니다.
    dummy_frames = [torch.randn(3, 81, 720, 1280, device=device)]
    dummy_masks = [torch.ones(1, 81, 720, 1280, device=device)]
    dummy_ref_images = [None]
    
    # CUDA 에러가 발생할 것이므로 try-except로 감싸서 프로그램이 죽지 않게 합니다.
    try:
        wan_vace.generate(
            input_prompt="a beautiful girl",
            input_frames=dummy_frames,
            input_masks=dummy_masks,
            input_ref_images=dummy_ref_images,
            sampling_steps=10, # 전체를 실행할 필요 없으므로 스텝 줄이기
            offload_model=False
        )
    except RuntimeError as e:
        print(f"\n--- CAUGHT EXPECTED ERROR ---")
        print(f"Error message: {e}")
        print("This is expected. Check the memory logs above.")
        log_memory(device, "At time of error")

if __name__ == "__main__":
    main()