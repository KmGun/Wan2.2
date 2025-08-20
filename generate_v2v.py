import argparse
import os
import logging
import sys
from PIL import Image
import imageio
import torch
from wan.video2video import CustomWanVace
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import save_video
from wan.annotators import PoseBodyFaceVideoAnnotator
from wan.annotators.utils import read_video_frames, save_one_video

def _parse_args():
    parser = argparse.ArgumentParser(description="Video to Video generation using Wan-VACE")
    # --- 경로 관련 인자 ---
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Wan2.2 베이스 모델 디렉토리 경로 (예: ./Wan2.2-T2V-A14B).")
    parser.add_argument("--vace-ckpt-path", type=str, required=True, help="Wan2.1-VACE 체크포인트 파일 경로 또는 디렉토리 (예: ./checkpoints/Wan2.1-VACE-1.3B/).")
    parser.add_argument("--vae-ckpt-path", type=str, default=None, help="VAE 체크포인트 파일 경로 (미지정시 베이스 모델의 기본 VAE 사용).")
    parser.add_argument("--input-video", type=str, required=True, help="변환할 원본 비디오 파일 경로.")
    parser.add_argument("--output-path", type=str, default="v2v_output.mp4", help="생성된 비디오를 저장할 경로.")

    # --- 생성 제어 관련 인자 ---
    parser.add_argument("--prompt", type=str, required=True, help="비디오 생성을 위한 텍스트 프롬프트.")
    parser.add_argument("--neg-prompt", type=str, default="", help="부정 프롬프트.")
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=['t2v-A14B', 'i2v-A14B', 'ti2v-5B'], help="사용할 Wan2.2 베이스 모델 태스크.")
    parser.add_argument("--sampling-steps", type=int, default=50, help="샘플링 스텝 수.")
    parser.add_argument("--context-scale", type=float, default=1.0, help="VACE 컨텍스트(구조)의 영향력. (기본값: 1.0)")
    parser.add_argument("--seed", type=int, default=-1, help="랜덤 시드.")
    parser.add_argument("--frame-num", type=int, default=81, help="생성할 비디오의 프레임 수. 4n+1 형태 권장.")
    parser.add_argument("--size", type=str, default="1280*720", help="생성될 비디오의 해상도.")

    # --- 선택적 VACE 입력 인자 ---
    parser.add_argument("--input-mask", type=str, default=None, help="[선택] 마스크 비디오 파일 경로 (제공 시 Inpainting 모드로 동작).")
    parser.add_argument("--ref-image", type=str, default=None, help="[선택] 참조 이미지 파일 경로.")

    # --- 시스템 관련 인자 ---
    parser.add_argument("--offload-model", action="store_true", help="메모리 절약을 위해 모델 오프로딩 활성화.")
    return parser.parse_args()

def main():
    args = _parse_args()
    print(f"[LOG] 인자 파싱 완료. 태스크: {args.task}, 프롬프트: {args.prompt}")

    # 기본 설정 불러오기
    config = WAN_CONFIGS[args.task]
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    print(f"[LOG] 설정 불러오기 완료. 샘플 FPS: {config.sample_fps}")

    # CustomWanVace 클래스 초기화
    print("[LOG] CustomWanVace 모델 초기화 시작...")
    wan_vace = CustomWanVace(
        config=config,
        wan2_2_ckpt_dir=args.ckpt_dir,
        wan2_1_vace_ckpt_path=args.vace_ckpt_path,
        vae_ckpt_path=args.vae_ckpt_path,
        device_id=0,
        init_on_cpu=True,
        offload_model=args.offload_model
    )
    print("[LOG] 모델 초기화 완료")

    # 스켈레톤 포즈 추출
    print("[LOG] 스켈레톤 포즈 추출 시작...")
    frames, *_ = read_video_frames(args.input_video, use_type='cv2', info=True)
    assert frames is not None, "Video read error"
    
    pose_annotator = PoseBodyFaceVideoAnnotator(
        cfg={
            "DETECTION_MODEL": "checkpoints/VACE-Annotators/pose/yolox_l.onnx",
            "POSE_MODEL": "checkpoints/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"
        },
        device=f'cuda:{0}'
    )
    pose_results = pose_annotator.forward(frames=frames)
    
    # 포즈 데이터 저장
    if pose_results:
        pose_output_path = os.path.join(os.path.dirname(args.output_path), 'pose_skeleton.mp4')
        os.makedirs(os.path.dirname(pose_output_path), exist_ok=True)
        save_one_video(pose_output_path, pose_results, fps=config.sample_fps)
        print(f"[LOG] 포즈 스켈레톤 비디오 저장 완료: {pose_output_path}")
    
    print("[LOG] 스켈레톤 포즈 추출 완료")


    print("[LOG] 소스 데이터 준비 시작...")
    size_key = args.size.strip("'\"“”")
    src_video, src_mask, src_ref_images = wan_vace.prepare_source([pose_output_path],
                                                                  [None],
                                                                  [None],
                                                                  args.frame_num, SIZE_CONFIGS[size_key], device)
    print("[LOG] 소스 데이터 준비 완료")


    # GENERATE!!
    print("[LOG] 비디오 생성 시작...")
    print(f"[LOG] 샘플링 스텝: {args.sampling_steps}, 컨텍스트 스케일: {args.context_scale}, 시드: {args.seed}")
    video_tensor = wan_vace.generate(
        args.prompt,
        src_video,
        src_mask,
        src_ref_images,
        size=SIZE_CONFIGS[size_key],
        frame_num=args.frame_num,
        offload_model=args.offload_model,
        n_prompt=args.neg_prompt,
        seed=args.seed,
    )
    print("[LOG] 비디오 생성 완료")

    # --- 강화된 비디오 저장 로직 --
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    if video_tensor is not None:
        logging.info(f"반환된 비디오 텐서 확인. Shape: {video_tensor.shape}, Dtype: {video_tensor.dtype}")
        try:
            logging.info(f"비디오 저장 시도: {args.output_path}")
            save_video(video_tensor, args.output_path, fps=config.sample_fps)
            logging.info(f"비디오 저장 완료: {args.output_path}")
        except Exception as e:
            logging.error(f"파일 저장 중 오류 발생: {args.output_path}", exc_info=True)
    else:
        logging.warning("생성 함수에서 반환된 비디오 텐서가 None입니다. 파일 저장을 건너뜁니다.")


if __name__ == "__main__":
    main()