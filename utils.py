import cv2
import os
from tqdm import tqdm

MIN_DETECTION_SIZE = 10

def load_frames(input_path):
    frames = []
    if os.path.isdir(input_path):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path)
                       if os.path.splitext(f)[1].lower() in image_extensions])
        for idx, file in enumerate(files):
            img = cv2.imread(file)
            if img is not None:
                frames.append({
                    "frame_id": idx,
                    "image": img,
                    "file_name": os.path.basename(file),
                    "detections": []
                })
    else:
        cap = cv2.VideoCapture(input_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append({
                "frame_id": idx,
                "image": frame,
                "file_name": f"frame_{idx:06d}.jpg",
                "detections": []
            })
            idx += 1
        cap.release()
    return frames

def run_yolo_detection(frames, model, classes, conf_thresh=0.25):
    print(f"Running YOLO detection with confidence threshold {conf_thresh}...")
    for frame_dict in tqdm(frames):
        results = model.predict(
            frame_dict["image"],
            conf=conf_thresh,
            verbose=False
        )
        detections = []
        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                xywh = box.xywh.cpu().numpy()[0]
                conf = float(box.conf.cpu().numpy()[0])
                cls_pred = int(box.cls.cpu().numpy()[0])
                if xywh[2] < MIN_DETECTION_SIZE or xywh[3] < MIN_DETECTION_SIZE:
                    continue
                detections.append({
                    "bbox": xywh.tolist(),
                    "class": cls_pred % len(classes),
                    "confidence": conf
                })
        frame_dict["detections"] = detections
    return frames

def filter_redundant_frames(frames, pos_thresh=100):
    print("Filtering redundant frames...")
    filtered = []
    last_detections = None
    for frame_dict in frames:
        current_dets = frame_dict["detections"]
        if last_detections is None or not detections_similar(last_detections, current_dets, pos_thresh):
            filtered.append(frame_dict)
            last_detections = current_dets
    print(f"Selected {len(filtered)} frames out of {len(frames)}")
    return filtered

def detections_similar(dets_a, dets_b, pos_thresh=100):
    """
    Checks if all detections in the smaller set have corresponding matches
    in the larger set within positional threshold.
    """
    # Handle empty detection cases
    if not dets_a and not dets_b:
        return True
    if not dets_a or not dets_b:
        return False
    
    # Find which detection set is smaller
    main_dets, comp_dets = (dets_a, dets_b) if len(dets_a) <= len(dets_b) else (dets_b, dets_a)
    
    # Create a copy to track unmatched comparisons
    comp_dets = comp_dets.copy()
    
    for md in main_dets:
        closest_distance = float('inf')
        closest_idx = -1
        
        # Find closest matching detection in comparison set
        for i, cd in enumerate(comp_dets):
            # Calculate center distance
            mx, my, mw, mh = md["bbox"]
            cx, cy, cw, ch = cd["bbox"]
            distance = ((mx - cx)**2 + (my - cy)**2)**0.5
            
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = i
        
        # Check if closest match is within threshold
        if closest_distance > pos_thresh:
            return False
        
        # Remove matched detection from comparison pool
        if closest_idx != -1:
            comp_dets.pop(closest_idx)
    
    return True

