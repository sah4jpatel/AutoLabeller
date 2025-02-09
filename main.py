#!/usr/bin/env python3
import argparse
from .utils import load_frames, run_yolo_detection, filter_redundant_frames
from .viewer import manual_verification
from .dataset import save_coco_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Auto Labeller Tool")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to video file or directory of image frames")
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                       help="List of class names (in desired order)")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to your finetuned YOLO model (e.g. model.pt)")
    parser.add_argument("--full_video", action="store_true",
                       help="If set, process all frames without redundancy filtering")
    parser.add_argument("--min_images", type=int, default=0,
                       help="Minimum number of images required in final dataset")
    parser.add_argument("--output", type=str, default="dataset.json",
                       help="Output COCO dataset filename")
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    return parser.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    
    frames = load_frames(args.input)
    print(f"Loaded {len(frames)} frames.")

    frames = run_yolo_detection(frames, model, args.classes, args.conf_thresh)

    if not args.full_video:
        frames = filter_redundant_frames(frames)
    
    frames = [f for f in frames if len(f["detections"]) > 0]
    print(f"{len(frames)} frames remain after filtering empty detection frames.")
    
    verified_frames = manual_verification(frames, args.classes)
    
    if args.min_images > 0 and len(verified_frames) < args.min_images:
        print(f"Warning: Only {len(verified_frames)} frames verified, "
              f"less than the requested minimum of {args.min_images}.")
    
    save_coco_dataset(verified_frames, args.classes, args.output)

if __name__ == "__main__":
    main()
