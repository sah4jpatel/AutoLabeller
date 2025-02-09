import os
import cv2
import json

def save_coco_dataset(frames, classes, output_file):
    output_dir = os.path.dirname(output_file)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    images = []
    annotations = []
    categories = []
    ann_id = 1
    img_id = 1

    for cid, cname in enumerate(classes):
        categories.append({
            "id": cid,
            "name": cname
        })
    
    for frame in frames:
        img_path = os.path.join(image_dir, frame["file_name"])
        cv2.imwrite(img_path, frame["image"])
        
        h, w = frame["image"].shape[:2]
        images.append({
            "id": img_id,
            "file_name": frame["file_name"],
            "height": h,
            "width": w
        })
        
        for det in frame["detections"]:
            x, y, bw, bh = det["bbox"]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": det["class"],
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1
        img_id += 1
    
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"COCO dataset saved to {output_file}")
