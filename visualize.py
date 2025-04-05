import os
import cv2


class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1
    MODE_TEST = 2
    
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.data_folder = os.path.join(dataset_folder, "data")
        
        # Load class names
        classes_file = os.path.join(dataset_folder, "classes.txt")
        with open(classes_file, "r") as f:
            self.classes = f.read().splitlines()
        self.classes = {i: c for i, c in enumerate(self.classes)}
        
        # Try different modes if one fails
        try:
            self.set_mode(YoloVisualizer.MODE_TEST)
            print("Using test dataset for visualization")
        except:
            try:
                self.set_mode(YoloVisualizer.MODE_VAL)
                print("Using validation dataset for visualization")
            except:
                try:
                    self.set_mode(YoloVisualizer.MODE_TRAIN)
                    print("Using training dataset for visualization")
                except Exception as e:
                    print(f"Error initializing visualizer: {e}")
                    raise
    
    def set_mode(self, mode=MODE_TRAIN):
        if mode == self.MODE_TRAIN:
            self.images_folder = os.path.join(self.data_folder, "train", "images")
            self.labels_folder = os.path.join(self.data_folder, "train", "labels")
        elif mode == self.MODE_VAL:
            self.images_folder = os.path.join(self.data_folder, "val", "images")
            self.labels_folder = os.path.join(self.data_folder, "val", "labels")
        else:  # TEST mode
            self.images_folder = os.path.join(self.data_folder, "test", "images")
            self.labels_folder = os.path.join(self.data_folder, "test", "labels")
        
        print(f"Looking for images in: {self.images_folder}")
        print(f"Looking for labels in: {self.labels_folder}")
        
        if not os.path.exists(self.images_folder):
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        if not os.path.exists(self.labels_folder):
            raise FileNotFoundError(f"Labels folder not found: {self.labels_folder}")
            
        self.image_names = sorted(os.listdir(self.images_folder))
        self.num_images = len(self.image_names)
        
        # Check if labels exist for all images
        self.label_names = []
        for img_name in self.image_names:
            # Convert image name to label name (change extension)
            base_name = os.path.splitext(img_name)[0]
            label_name = f"{base_name}.txt"
            if os.path.exists(os.path.join(self.labels_folder, label_name)):
                self.label_names.append(label_name)
        
        print(f"Found {self.num_images} images and {len(self.label_names)} labels")
        
        if self.num_images == 0:
            raise ValueError("No images found in the specified folder")
            
        self.frame_index = 0

    def next_frame(self):
        self.frame_index += 1
        if self.frame_index >= self.num_images:
            self.frame_index = 0
        elif self.frame_index < 0:
            self.frame_index = self.num_images - 1

    def previous_frame(self):
        self.frame_index -= 1
        if self.frame_index >= self.num_images:
            self.frame_index = 0
        elif self.frame_index < 0:
            self.frame_index = self.num_images - 1
    
    def seek_frame(self, idx):
        image_file = os.path.join(self.images_folder, self.image_names[idx])
        image = cv2.imread(image_file)
        if image is None:
            print(f"Could not read image: {image_file}")
            return None
        
        # Get corresponding label filename
        base_name = os.path.splitext(self.image_names[idx])[0]
        label_file = os.path.join(self.labels_folder, f"{base_name}.txt")
        
        # Draw bounding boxes if label file exists
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.read().splitlines()
                
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    class_index, x, y, w, h = map(float, parts[:5])
                    cx = int(x * image.shape[1])
                    cy = int(y * image.shape[0])
                    w = int(w * image.shape[1])
                    h = int(h * image.shape[0])
                    x = cx - w // 2
                    y = cy - h // 2
                    
                    # Draw bounding box and label
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    class_name = self.classes.get(int(class_index), f"Class {int(class_index)}")
                    cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Draw info about missing label
            cv2.putText(image, "No labels for this image", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add image info
        cv2.putText(image, f"Image {idx+1}/{self.num_images}: {self.image_names[idx]}", 
                   (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image

    def run(self):
        print("Visualizer controls:")
        print("  'a' - Previous image")
        print("  'd' - Next image")
        print("  't' - Switch to training set")
        print("  'v' - Switch to validation set")
        print("  'x' - Switch to test set")
        print("  'q' or ESC - Quit")
        
        while True:
            try:
                frame = self.seek_frame(self.frame_index)
                if frame is None:
                    print(f"Skipping invalid frame {self.frame_index}")
                    self.next_frame()
                    continue
                    
                # Resize large images while maintaining aspect ratio
                height, width = frame.shape[:2]
                max_height = 900
                max_width = 1600
                if height > max_height or width > max_width:
                    scale = min(max_height / height, max_width / width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imshow(f"YOLOv8 Visualizer - {os.path.basename(self.images_folder)}", frame)
                key = cv2.waitKey(0)
                
                if key == ord('q') or key == 27 or key == -1:  # q or ESC
                    break
                elif key == ord('d'):  # next
                    self.next_frame()
                elif key == ord('a'):  # previous
                    self.previous_frame()
                elif key == ord('t'):  # train
                    self.set_mode(YoloVisualizer.MODE_TRAIN)
                elif key == ord('v'):  # val
                    self.set_mode(YoloVisualizer.MODE_VAL)
                elif key == ord('x'):  # test
                    self.set_mode(YoloVisualizer.MODE_TEST)
            except Exception as e:
                print(f"Error processing frame: {e}")
                self.next_frame()
                
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        vis = YoloVisualizer(os.path.dirname(os.path.abspath(__file__)))
        vis.run()
    except Exception as e:
        print(f"Error: {e}")
