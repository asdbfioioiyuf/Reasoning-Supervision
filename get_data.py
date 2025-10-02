"""
Data preprocessing utilities for video action recognition
Extracts frames and generates annotations for spatial attention supervision
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


# Configuration
GRID_H, GRID_W = 4, 4
FRAMES_PER_VIDEO = 10  # Extract equally distributed frames


def extract_frames_from_video(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
    """
    Extract equally distributed frames from a video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames

    # Calculate frame indices
    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        step = total_frames // num_frames
        frame_indices = [i * step for i in range(num_frames)]

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def create_grid_visualization(frame: np.ndarray, grid_h: int = 4, grid_w: int = 4, spacing: int = 5) -> Image.Image:
    """
    Create a grid visualization of the frame.

    Args:
        frame: Input frame as numpy array
        grid_h, grid_w: Grid dimensions
        spacing: Spacing between grid cells

    Returns:
        PIL Image with grid overlay
    """
    img = Image.fromarray(frame)
    width, height = img.size
    cell_w = width // grid_w
    cell_h = height // grid_h

    out_w = cell_w * grid_w + spacing * (grid_w - 1)
    out_h = cell_h * grid_h + spacing * (grid_h - 1)
    grid_image = Image.new('RGB', (out_w, out_h), color='white')

    for r in range(grid_h):
        for c in range(grid_w):
            l = c * cell_w
            t = r * cell_h
            rgt = l + cell_w
            btm = t + cell_h
            cell = img.crop((l, t, rgt, btm))

            x = c * (cell_w + spacing)
            y = r * (cell_h + spacing)
            grid_image.paste(cell, (x, y))

    return grid_image


def process_video_annotations(video_path: str, annotation_data: Dict, output_dir: str, class_name: str):
    """
    Process video and extract frame-level annotations.

    Args:
        video_path: Path to video file
        annotation_data: Annotation dictionary
        output_dir: Output directory for processed data
        class_name: Action class name

    Returns:
        Dictionary with processed annotations
    """
    frames = extract_frames_from_video(video_path, num_frames=FRAMES_PER_VIDEO)
    if not frames:
        return {}

    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]

    # Create output directories
    video_output_dir = Path(output_dir) / class_name / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    processed_annotations = {}

    for frame_idx, frame in enumerate(frames):
        # Save frame
        frame_path = video_output_dir / f"frame_{frame_idx:04d}.jpg"
        Image.fromarray(frame).save(frame_path, quality=95)

        # Create grid visualization
        grid_img = create_grid_visualization(frame, GRID_H, GRID_W)
        grid_path = video_output_dir / f"frame_{frame_idx:04d}_grid.jpg"
        grid_img.save(grid_path, quality=95)

        # Store frame info
        frame_data = {
            "frame_path": str(frame_path),
            "grid_path": str(grid_path),
            "class": class_name
        }

        # Add annotations if available
        if annotation_data and str(frame_idx) in annotation_data:
            frame_data["annotation"] = annotation_data[str(frame_idx)]

        processed_annotations[str(frame_idx)] = frame_data

    return {video_id: processed_annotations}


def get_all_video_files(dataset_dir: str) -> Dict[str, List[str]]:
    """
    Scan dataset directory and group videos by class.

    Args:
        dataset_dir: Root dataset directory

    Returns:
        Dictionary mapping class names to lists of video paths
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return {}

    videos_by_class = {}

    # Scan for subdirectories (each subdirectory is a class)
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        video_files = []

        # Find all video files in class directory
        for ext in ['*.avi', '*.mp4', '*.mkv', '*.mov']:
            video_files.extend(class_dir.glob(ext))

        if video_files:
            videos_by_class[class_name] = [str(vf) for vf in video_files]

    return videos_by_class


def main():
    parser = argparse.ArgumentParser(description='Process video dataset for action recognition')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='Output directory')
    parser.add_argument('--anno_file', type=str, default=None, help='Path to annotations JSON file')
    parser.add_argument('--max_videos_per_class', type=int, default=None, help='Maximum videos to process per class')

    args = parser.parse_args()

    # Load annotations if provided
    annotations = {}
    if args.anno_file and os.path.exists(args.anno_file):
        with open(args.anno_file, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded annotations from {args.anno_file}")

    # Get all videos
    videos_by_class = get_all_video_files(args.dataset_dir)
    if not videos_by_class:
        print("[ERROR] No videos found in dataset directory")
        return

    print(f"Found {len(videos_by_class)} classes")
    for class_name, videos in videos_by_class.items():
        print(f"  {class_name}: {len(videos)} videos")

    # Process videos
    all_processed = {}

    for class_name, video_paths in tqdm(videos_by_class.items(), desc="Processing classes"):
        # Limit videos per class if specified
        if args.max_videos_per_class:
            video_paths = video_paths[:args.max_videos_per_class]

        for video_path in tqdm(video_paths, desc=f"  {class_name}", leave=False):
            video_id = Path(video_path).stem

            # Get annotations for this video if available
            video_anno = annotations.get(video_id, {})

            # Process video
            try:
                processed = process_video_annotations(
                    video_path=video_path,
                    annotation_data=video_anno,
                    output_dir=args.output_dir,
                    class_name=class_name
                )
                all_processed.update(processed)
            except Exception as e:
                print(f"[ERROR] Failed to process {video_path}: {e}")
                continue

    # Save processed annotations
    output_anno_path = Path(args.output_dir) / "processed_annotations.json"
    with open(output_anno_path, 'w') as f:
        json.dump(all_processed, f, indent=2)

    print(f"\n Processing completed!")
    print(f"   Processed {len(all_processed)} videos")
    print(f"   Annotations saved to: {output_anno_path}")


if __name__ == "__main__":
    main()
