import cv2
import numpy as np
from screeninfo import get_monitors

class FrameViewer:
    def __init__(self, classes):
        self.classes = classes
        self.window_name = "Dataset Labeler"
        self.current_frame = None
        self.detections = []
        self.dragging = False
        self.start_point = None
        self.scale_factor = 1.0
        self.window_initialized = False
        
        # Set up window once
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_event)
        
    def update_frame(self, frame_dict):
        """Update the current frame and detections without recreating window"""
        self.original_frame = frame_dict["image"].copy()
        self.current_frame = frame_dict
        self.detections = frame_dict["detections"].copy()
        
        # Calculate window size only once
        if not self.window_initialized:
            monitor = get_monitors()[0]
            target_height = int(monitor.height * 0.75)
            h, w = self.original_frame.shape[:2]
            self.scale_factor = target_height / h
            target_width = int(w * self.scale_factor)
            
            cv2.resizeWindow(self.window_name, target_width, int(target_height))
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            self.window_initialized = True
            
        self.redraw()

    def redraw(self):
        display_frame = self.original_frame.copy()
        
        self.scaled_frame = cv2.resize(display_frame, None, fx=self.scale_factor, fy=self.scale_factor)
        
        # Draw detections
        for det in self.detections:
            label = self.classes[det["class"]]
            xc, yc, w, h = det["bbox"]
            x1 = int((xc - w/2) * self.scale_factor)
            y1 = int((yc - h/2) * self.scale_factor)
            x2 = int((xc + w/2) * self.scale_factor)
            y2 = int((yc + h/2) * self.scale_factor)
            cv2.rectangle(self.scaled_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.scaled_frame, f"{label}:{det['confidence']:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add info text
        info_text = ("Left-click: cycle class | Right-click: remove | "
                    f"Detections: {len(self.detections)} | Frame: {self.current_frame['frame_id']}")
        cv2.putText(self.scaled_frame, info_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, self.scaled_frame)

    def mouse_event(self, event, x, y, flags, param):
        # Convert coordinates to original scale
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
    
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is inside any existing box
            for det in self.detections:
                xc, yc, w, h = det["bbox"]
                x1, y1 = int(xc - w/2), int(yc - h/2)
                x2, y2 = int(xc + w/2), int(yc + h/2)
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    det["class"] = (det["class"] + 1) % len(self.classes)
                    self.redraw()
                    break
            else:  # No box was clicked
                self.dragging = True
                self.start_point = (orig_x, orig_y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            temp_frame = self.scaled_frame.copy()
            start_scaled = (int(self.start_point[0]),int(self.start_point[1]))
            cv2.rectangle(temp_frame, start_scaled, (x, y), (255, 0, 0), 2)
            cv2.imshow(self.window_name, temp_frame)
            
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start_point
            x2, y2 = int(orig_x), int(orig_y)
            
            # Only add box if area is greater than minimum (40x40 pixels)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w * h > 400:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                self.detections.append({
                    "bbox": [cx, cy, w, h],
                    "class": 0,
                    "confidence": 0.99
                })
            self.redraw()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, det in enumerate(self.detections):
                xc, yc, w, h = det["bbox"]
                x1, y1 = int(xc - w/2), int(yc - h/2)
                x2, y2 = int(xc + w/2), int(yc + h/2)
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    self.detections.pop(i)
                    self.redraw()
                    break
                
        # After any changes, redraw
        self.redraw()

def manual_verification(frames, classes):
    print("\nStarting manual verification:")
    print(" • Left click: cycle class | Right click: delete")
    print(" • Click+drag: draw new box | SPACE/ENTER: accept")
    print(" • 'd': discard frame | 'q': quit\n")
    
    viewer = FrameViewer(classes)
    verified_frames = []
    index = 0
    
    while index < len(frames) and cv2.getWindowProperty(viewer.window_name, cv2.WND_PROP_VISIBLE) >= 1:
        frame = frames[index]
        viewer.update_frame(frame)
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key in (32, 13):  # Space/Enter
                frame["detections"] = viewer.detections
                verified_frames.append(frame)
                index += 1
                break
            elif key == ord('d'):
                print(f"Frame {frame['frame_id']} discarded.")
                index += 1
                break
            elif key == ord('q'):
                index = len(frames)
                break
                
    cv2.destroyAllWindows()
    return verified_frames
