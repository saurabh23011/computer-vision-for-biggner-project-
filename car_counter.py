import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import os

class CarCounter:
    def __init__(self, video_path, model_path, mask_path, output_path="output.mp4"):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.model = YOLO(model_path)
        
        # Load and check mask
        self.mask = cv2.imread(mask_path)
        if self.mask is None:
            raise ValueError(f"Could not load mask image: {mask_path}")
        
        # Initialize tracker
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.limits = [400, 297, 673, 297]
        self.total_count = []
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        # Resize mask to match video dimensions
        self.mask = cv2.resize(self.mask, (self.frame_width, self.frame_height))
        
        # Define target classes for vehicle detection
        self.target_classes = {"car", "truck", "bus", "motorbike"}
        
        # Load class names
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]

    def preprocess_mask(self, frame):
        """Ensure mask matches frame dimensions and channels"""
        if self.mask.shape[:2] != frame.shape[:2]:
            self.mask = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
        
        if len(self.mask.shape) == 2:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        if len(self.mask.shape) != len(frame.shape):
            if len(frame.shape) == 3:
                self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        return self.mask

    def process_frame(self, img):
        if img is None:
            return None
            
        # Preprocess mask to match frame
        mask = self.preprocess_mask(img)
        
        # Apply mask to region of interest
        img_region = cv2.bitwise_and(img, mask)
        
        # Get YOLO detections
        results = self.model(img_region, stream=True)
        detections = np.empty((0, 5))
        
        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]
                
                if current_class in self.target_classes and conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))
        
        # Update tracker
        results_tracker = self.tracker.update(detections)
        
        # Draw counting line
        cv2.line(img, (self.limits[0], self.limits[1]), 
                (self.limits[2], self.limits[3]), (0, 0, 255), 5)
        
        # Process tracking results
        for result in results_tracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w, h = x2 - x1, y2 - y1
            
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                             scale=2, thickness=3, offset=10)
            
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            if (self.limits[0] < cx < self.limits[2] and 
                self.limits[1] - 15 < cy < self.limits[1] + 15):
                if id not in self.total_count:
                    self.total_count.append(id)
                    cv2.line(img, (self.limits[0], self.limits[1]), 
                            (self.limits[2], self.limits[3]), (0, 255, 0), 5)
        
        # Display count
        cv2.putText(img, str(len(self.total_count)), (255, 100),
                   cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
        
        return img

    def run(self):
        frame_count = 0
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Finished processing video")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    print(f"Processing frame {frame_count}")
                
                processed_img = self.process_frame(img)
                if processed_img is None:
                    print(f"Failed to process frame {frame_count}")
                    continue
                
                # Write the frame to output video
                self.out.write(processed_img)
                
        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            # Clean up
            self.cap.release()
            self.out.release()
            print(f"Processing complete. Output saved to {self.output_path}")
            print(f"Total vehicles counted: {len(self.total_count)}")

# Usage
if __name__ == "__main__":
    try:
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        counter = CarCounter(
            video_path="Videos/cars.mp4",
            model_path="yolo11l.pt",
            mask_path="mask.png",
            output_path=os.path.join(output_dir, "output.mp4")
        )
        counter.run()
    except Exception as e:
        print(f"Error running car counter: {e}")