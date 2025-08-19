import argparse
import os
import sys
import torch
import numpy as np

# CUDA 관련 에러 우회를 위한 monkey patching
if not torch.cuda.is_available():
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 0
    torch.cuda.is_available = lambda: False

from wan.annotators import PoseBodyFaceVideoAnnotator
from wan.annotators.utils import read_video_frames, save_one_video

def _parse_args():
    parser = argparse.ArgumentParser(description="Pose extraction test script")
    parser.add_argument("--input-video", type=str, required=True, help="변환할 원본 비디오 파일 경로.")
    parser.add_argument("--output-path", type=str, default="pose_skeleton.mp4", help="포즈 스켈레톤 비디오를 저장할 경로.")
    parser.add_argument("--fps", type=float, default=24.0, help="출력 비디오의 FPS.")
    parser.add_argument("--device", type=str, default="cpu", help="사용할 디바이스 (cpu 등).")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    print(f"[LOG] 비디오 읽기 시작: {args.input_video}")
    frames, *info = read_video_frames(args.input_video, use_type='cv2', info=True)
    
    if frames is None:
        print("[ERROR] 비디오를 읽을 수 없습니다.")
        sys.exit(1)
    
    print(f"[LOG] 비디오 정보: {len(frames)} 프레임")
    original_fps = args.fps
    if info:
        print(f"[LOG] 추가 정보: {info}")
        if len(info) > 0 and info[0] is not None:
            original_fps = info[0]
            print(f"[LOG] 원본 FPS 감지: {original_fps}")
    
    print("[LOG] 포즈 어노테이터 초기화...")
    pose_annotator = PoseBodyFaceVideoAnnotator(
        cfg={
            "DETECTION_MODEL": "checkpoints/VACE-Annotators/pose/yolox_l.onnx",
            "POSE_MODEL": "checkpoints/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"
        },
        device=args.device
    )
    
    print("[LOG] 포즈 추출 시작...")
    pose_results = pose_annotator.forward(frames=frames)
    
    if pose_results is not None:
        if isinstance(pose_results, dict) and 'frames' in pose_results:
            os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
            save_one_video(args.output_path, pose_results['frames'], fps=original_fps)
            print(f"[LOG] 포즈 스켈레톤 비디오 저장 완료: {args.output_path}")
            print(f"[LOG] 출력 프레임 수: {len(pose_results['frames'])}, FPS: {original_fps}")
        elif isinstance(pose_results, (list, np.ndarray)):
            os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
            save_one_video(args.output_path, pose_results, fps=original_fps)
            print(f"[LOG] 포즈 스켈레톤 비디오 저장 완료: {args.output_path}")
            print(f"[LOG] 출력 프레임 수: {len(pose_results)}, FPS: {original_fps}")
        else:
            print(f"[ERROR] 예상치 못한 결과 형식: {type(pose_results)}")
    else:
        print("[ERROR] 포즈 추출 실패 또는 결과가 없습니다.")

if __name__ == "__main__":
    main()