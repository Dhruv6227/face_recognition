import cv2
import argparse
import os 

def process_img(img):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Blur each detected face
    for (x, y, w, h) in faces:
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        # Apply blur to face region
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30,30))
    
    return img

def main():
    args = argparse.ArgumentParser()    
    args.add_argument("--input", type=str, default="webcam")
    args.add_argument("--filepath", default=None)
    args = args.parse_args()

    if args.input == "image":
        if args.filepath is None:
            print("Error: Please provide filepath for image input")
            return
            
        img = cv2.imread(args.filepath)
        if img is not None:
            img = process_img(img)
            cv2.imwrite("output_image.jpg", img)
            print("Processed image saved as output_image.jpg")
        else:
            print(f"Error: Could not read image from {args.filepath}")
            
    elif args.input == "video":
        if args.filepath is None:
            print("Error: Please provide filepath for video input")
            return
            
        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()
        if not ret:
            print("Error reading video file")
            return
            
        H, W = frame.shape[:2]
        # Use cv2.VideoWriter with integer FourCC code instead
        fourcc = 0x7634706d  # This is equivalent to 'mp4v'
        output_video = cv2.VideoWriter('output.mp4', 
                                     fourcc, 
                                     30,
                                     (W, H))
        
        while ret:
            frame = process_img(frame)
            output_video.write(frame)
            ret, frame = cap.read()
            
        cap.release()
        output_video.release()
        print("Processed video saved as output.mp4")
        
    elif args.input == 'webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = process_img(frame)
            cv2.imshow('Face Blur (Press Q to quit)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
