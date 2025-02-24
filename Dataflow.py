import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv5 Object Detection Model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MiDaS Depth Estimation Model
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()

# Load MiDaS Transformations
transform_pipeline = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Create output folder for processed images
output_folder = "./output_images"
os.makedirs(output_folder, exist_ok=True)

def detect_pedestrians(image_path):
    """Detects pedestrians in an image and estimates their depth."""
    
    # Load and preprocess input image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run YOLOv5 detection
    detection_results = yolo_model(img_rgb, size=1024)
    detections = detection_results.pandas().xyxy[0]
    
    # Filter out only pedestrian detections
    pedestrians = detections[detections['name'] == 'person']
    
    # Estimate depth using MiDaS
    resized_img = cv2.resize(img_rgb, (256, 256))
    transformed_img = transform_pipeline(resized_img).to(torch.device('cpu'))
    
    with torch.no_grad():
        depth_map = depth_model(transformed_img)
    
    # Convert depth map to numpy array and resize back to original image size
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_resized = cv2.resize(depth_map, (img.shape[1], img.shape[0]))
    
    # Extract pedestrian bounding boxes, confidence scores, and depth values
    results = []
    for _, row in pedestrians.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = float(row['confidence'])
        
        # Calculate average depth within bounding box
        avg_depth = np.mean(depth_resized[y_min:y_max, x_min:x_max])
        
        results.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": round(confidence, 2),
            "distance": round(avg_depth, 2)
        })
    
    return results

def process_images(folder_path):
    """Processes images in a given folder that start with 'A' or 'C'."""
    
    for file_name in os.listdir(folder_path):
        if (file_name.startswith('A') or file_name.startswith('C')) and file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing image: {file_name}")
            
            # Run pedestrian detection
            detections = detect_pedestrians(image_path)
            print(f"Detections for {file_name}: {detections}")
            
            # Load image to draw detections
            img = cv2.imread(image_path)
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                distance = detection['distance']
                
                # Draw bounding box
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Add label
                label = f"Person {confidence}, {distance}m"
                cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the processed image
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
            
            # Display the image
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Processed: {file_name}")
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    dataset_folder = "./Dataset_Occluded_Pedestrian/"
    process_images(dataset_folder)
