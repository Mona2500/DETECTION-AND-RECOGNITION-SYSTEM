import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
import cv2 as cv
import face_recognition
import os
import numpy as np
from PIL import Image, ImageTk
import time
from ultralytics import YOLO
import easyocr
from paddleocr import PaddleOCR
import math
import cvzone
import pickle 

PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ACCENT_COLOR = "#e74c3c"
LIGHT_COLOR = "#ecf0f1"
FONT_NAME = "Times New Roman"
FONT_SIZE_TITLE = 24
FONT_SIZE_HEADER = 18
FONT_SIZE_BODY = 12

window_stack = []

yolo_model = YOLO('../Yolo_Weights/yolov8l.pt')
plate_cascade = cv.CascadeClassifier("C:/Users/Asus/py_project/Object_Detection/NumberplateSample/haarcascade_plate_number.xml")
reader = easyocr.Reader(['en'])
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def open_window(new_window_func, *args):
    if window_stack:
        current = window_stack[-1]
        current.withdraw() 
    win = new_window_func(*args)
    window_stack.append(win)

def go_back():
    if len(window_stack) > 1:
        current = window_stack.pop()
        current.destroy()
        window_stack[-1].deiconify()

def show_about():
    about_win = tk.Toplevel()
    about_win.title("About Detection System")
    about_win.iconbitmap(r'Assets/pattern.png.ico')
    about_win.geometry("500x300")
    about_win.resizable(False, False)
    about_win.configure(bg=PRIMARY_COLOR)
    
    tk.Label(about_win, text="Detection System", font=(FONT_NAME, FONT_SIZE_TITLE, "bold"),
             bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=20)
    
    about_text = (
        "This application provides advanced computer vision capabilities.\n\n"
        "Features include:\n"
        "- Real-time object detection from webcam\n"
        "- Face detection and recognition\n"
        "- License plate detection and recognition (ANPR)\n"
        "- Support for both images and videos\n\n"
        "Developed using Python, OpenCV, YOLOv8, and Tkinter."
    )
    
    tk.Label(about_win, text=about_text, font=(FONT_NAME, FONT_SIZE_BODY),
             bg=PRIMARY_COLOR, fg=LIGHT_COLOR, justify=tk.LEFT).pack(pady=10)
    
    tk.Button(about_win, text="Close", command=about_win.destroy,
              font=(FONT_NAME, FONT_SIZE_HEADER), bg=ACCENT_COLOR, fg=LIGHT_COLOR).pack(pady=20)

def show_help():
    help_win = tk.Toplevel()
    help_win.title("Help Center")
    help_win.iconbitmap(r'Assets/pattern.png.ico')
    help_win.geometry("600x600")
    help_win.resizable(False, False)
    help_win.configure(bg=PRIMARY_COLOR)
    
    tk.Label(help_win, text="Help Center", font=(FONT_NAME, FONT_SIZE_TITLE, "bold"),
             bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=20)
    
    help_text = (
        "How to use the Detection System:\n"
        "1. Object Detection:\n"
        "   - Click to start real-time object detection from webcam\n\n"
        "2. License Plate Detection:\n"
        "   - Select 'From Image' to detect plates in an image\n"
        "   - Select 'From Video' to detect plates in a video file\n\n"
        "3. ANPR (Automatic Number Plate Recognition):\n"
        "   - Detects and reads license plates from images/videos\n\n"
        "4. Face Detection:\n"
        "   - Detects faces in images or via webcam\n\n"
        "5. Face Recognition:\n"
        "   - Recognizes known faces from your 'Images' folder\n\n"
        "Controls:\n"
        "- Press 'Q' to quit any video detection window\n"
        "- Press SPACE to pause detection in some modes\n"
        "- Use the 'Back' button to return to previous screens\n\n"
        "For technical support, please contact: support@detectionsystem.com"
    )

    tk.Label(help_win, text=help_text, font=(FONT_NAME, FONT_SIZE_BODY),
             bg=PRIMARY_COLOR, fg=LIGHT_COLOR, justify=tk.LEFT).pack(pady=10)
    
    tk.Button(help_win, text="Close", command=help_win.destroy,
              font=(FONT_NAME, FONT_SIZE_HEADER), bg=ACCENT_COLOR, fg=LIGHT_COLOR).pack(pady=20)

def load_known_faces(path='Images'):
    pickle_path = os.path.join(path, "C:/Users/Asus/Documents/INTERNSHIP/PYTHON_PRAC/encodings.pickle")
    
    # If pickle file exists and is recent, use it
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                known_data = pickle.load(f)
                print(f"Loaded {len(known_data['names'])} face encodings from pickle file")
                return known_data['encodings'], known_data['names']
        except Exception as e:
            print(f"Error loading pickle file: {e}. Rebuilding face database...")
    
    known_encodings = []
    known_names = []

    for filename in os.listdir(path):
        if filename == 'known_faces.pkl':
            continue
            
        full_path = os.path.join(path, filename)
        try:
            image = face_recognition.load_image_file(full_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                # Use the first face found in the image
                known_encodings.append(encodings[0])
                # Remove file extension and any numbers/dates for cleaner names
                name = os.path.splitext(filename)[0].split('_')[0].title()
                known_names.append(name)
                print(f"Processed {filename} as {name}")
            else:
                print(f"[WARNING] No face found in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the encodings to pickle file for faster loading next time
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
        print(f"Saved {len(known_names)} face encodings to pickle file")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

    return known_encodings, known_names

def update_known_faces():
    global encodings, names
    encodings, names = load_known_faces('Images')
    messagebox.showinfo("Success", "Known faces database updated successfully!")

def object_detection_webcam():
    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, vid = cap.read()
        results = yolo_model(vid, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(vid, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv.imshow('Object Detection', vid)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def license_plate_detection_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    img = cv.imread(file_path)
    if img is None:
        return
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(img_pil)
    
    result_window = tk.Toplevel()
    result_window.title("License Plate Detection Result")
    result_window.configure(bg=PRIMARY_COLOR)
    
    tk.Label(result_window, text="License Plate Detection Result", 
             font=(FONT_NAME, FONT_SIZE_HEADER, "bold"),
             bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)
    
    label = tk.Label(result_window, image=img_tk, bg=PRIMARY_COLOR)
    label.image = img_tk
    label.pack()
    
    save_button = tk.Button(result_window, text="Save Result", 
                           command=lambda: save_image_result(img_rgb),
                           font=(FONT_NAME, FONT_SIZE_BODY),
                           bg=SECONDARY_COLOR, fg=LIGHT_COLOR)
    save_button.pack(pady=10)

def save_image_result(image):
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                           filetypes=[("JPEG files", "*.jpg"),
                                                      ("PNG files", "*.png")])
    if file_path:
        cv.imwrite(file_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
        messagebox.showinfo("Success", "Image saved successfully!")

def license_plate_detection_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if not video_path:
        return

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video {video_path}")
        return

    cv.namedWindow("License Plate Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("License Plate Detection", 800, 600)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_resize = cv.resize(frame, (800, 600))
        gray = cv.cvtColor(frame_resize, cv.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in plates:
            cv.rectangle(frame_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame_resize, 'License Plate', (x, y - 5), cv.FONT_HERSHEY_COMPLEX,
                       0.5, (0, 0, 255), 1)

        cv.imshow("License Plate Detection", frame_resize)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def anpr_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
   
    original_image = cv.imread(file_path)
    if original_image is None:
        return
   
    def preprocess_for_ocr(image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed1 = clahe.apply(gray)
    
        processed2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY, 11, 2)
      
        _, processed3 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        return processed1, processed2, processed3
   
    def find_plates(image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(blurred, 50, 200)
        
        contours, _ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
        
        plates = []
        for contour in contours:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 2.5 < aspect_ratio < 5.5 and w > 80:
                    plates.append((x, y, w, h))
        
        return plates
    
    potential_plates = find_plates(original_image)
    
    best_text = ""
    best_plate = None
    max_confidence = 0
    
    for (x, y, w, h) in potential_plates:
        plate_region = original_image[y:y+h, x:x+w]
       
        preprocessed_variants = preprocess_for_ocr(plate_region)
        
        for i, plate in enumerate(preprocessed_variants):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv.filter2D(plate, -1, kernel)
       
            if h < 40:
                plate = cv.resize(plate, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
       
            for ocr_engine in ['easyocr', 'paddleocr']:
                try:
                    if ocr_engine == 'easyocr':
                        results = reader.readtext(plate, detail=0)
                        if results:
                            text = " ".join(results).strip().upper()
                            confidence = 0.8  
                    else:
                        results = paddle_ocr.ocr(plate, cls=True)
                        if results and results[0]:
                            text = " ".join([line[1][0] for line in results[0]]).strip().upper()
                            confidence = sum([line[1][1] for line in results[0]])/len(results[0]) if results[0] else 0
                   
                    if len(text) >= 4 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_text = text
                            best_plate = (x, y, w, h)
                except Exception as e:
                    print(f"OCR Error ({ocr_engine}): {e}")
   
    if best_plate:
        x, y, w, h = best_plate
        cv.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(original_image, best_text, (x, y-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
      
        img_rgb = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        result_window = tk.Toplevel()
        result_window.title("ANPR Result")
        result_window.configure(bg=PRIMARY_COLOR)
        
        tk.Label(result_window, text="ANPR Result", 
                font=(FONT_NAME, FONT_SIZE_HEADER, "bold"),
                bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)
        
        label = tk.Label(result_window, image=img_tk, bg=PRIMARY_COLOR)
        label.image = img_tk
        label.pack()
        
        tk.Button(result_window, text="Save Result", 
                command=lambda: save_image_result(img_rgb),
                font=(FONT_NAME, FONT_SIZE_BODY),
                bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)
    else:
        messagebox.showinfo("Info", "No license plate detected or recognized")

def preprocess_plate(plate_image):
    plate_resized = cv.resize(plate_image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(plate_resized, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv.filter2D(enhanced, -1, sharpen_kernel)
    plate_rgb = cv.cvtColor(sharpened, cv.COLOR_GRAY2RGB)
    return plate_rgb

def anpr_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if not video_path:
        return

    model_path = "../Yolo_Weights/yolov8n.pt"
    model = YOLO(model_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video {video_path}")
        return

    frame_skip = 1
    frame_count = 0
    line_y = 300
    processed_plates = []
    detected_plates = []

    cv.namedWindow("Plate Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("Plate Detection", 640, 480)

    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = cv.resize(frame.copy(), (640, 480))
        process_frame = cv.resize(frame, (640, 480))
        cv.line(display_frame, (0, line_y), (640, line_y), (0, 0, 255), 2)

        frame_count += 1
        process_this_frame = frame_count % frame_skip == 0

        if process_this_frame:
            results = model(process_frame)

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_crop = process_frame[y1:y2, x1:x2]
                    plate_center_y = (y1 + y2) // 2
                    plate_id = f"{x1}-{y1}-{x2}-{y2}"

                    if plate_center_y >= line_y and plate_id not in processed_plates:
                        try:
                            preprocessed_img = preprocess_plate(plate_crop)
                            ocr_results = ocr.ocr(preprocessed_img, cls=True)

                            if ocr_results and ocr_results[0]:
                                plate_text = " ".join([line[1][0] for line in ocr_results[0]])
                                print(f"Detected: {plate_text}")
                                processed_plates.append(plate_id)
                                detected_plates.append(plate_text)

                                cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv.putText(display_frame, plate_text, (x1, y1 - 10),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"OCR error: {e}")
                    else:
                        cv.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        current_time = time.time()
        elapsed = (current_time - last_frame_time) * 1000

        cv.imshow("Plate Detection", display_frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    if detected_plates:
        with open("detected_plates.txt", "w") as f:
            for plate in detected_plates:
                f.write(plate + "\n")
        messagebox.showinfo("Detection Complete", f"{len(detected_plates)} plates saved to detected_plates.txt")
    else:
        messagebox.showinfo("Detection Complete", "No plates were detected.")


def face_detection_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = cv.imread(file_path)
    if image is None:
        return
    
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image, model="cnn")

    for (top, right, bottom, left) in face_locations:
        cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    display_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_image)
    img_pil.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(img_pil)
    
    result_window = tk.Toplevel()
    result_window.title("Face Detection Result")
    result_window.configure(bg=PRIMARY_COLOR)
    
    tk.Label(result_window, text="Face Detection Result", 
            font=(FONT_NAME, FONT_SIZE_HEADER, "bold"),
            bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)
    
    label = tk.Label(result_window, image=img_tk, bg=PRIMARY_COLOR)
    label.image = img_tk
    label.pack()
    
    tk.Button(result_window, text="Save Result", 
             command=lambda: save_image_result(display_image),
             font=(FONT_NAME, FONT_SIZE_BODY),
             bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

def face_detection_webcam():
    cap = cv.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        cap_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_location = face_recognition.face_locations(cap_rgb, model="cnn")

        for (top, right, bottom, left) in face_location:
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv.imshow('Face Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def face_recognition_image(known_encodings, known_names):
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = cv.imread(file_path)
    if image is None:
        return
    
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb, model="hog")
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        name = "Unknown"
        confidence = "0%"
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                confidence = f"{round((1 - face_distances[best_match_index]) * 100, 2)}%"
       
        cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        text = f"{name} {confidence}"
        cv.putText(image, text, (left, bottom + 25), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    
    display_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_image)
    img_pil.thumbnail((800, 700))
    img_tk = ImageTk.PhotoImage(img_pil)
    
    result_window = tk.Toplevel()
    result_window.title("Face Recognition Result")
    result_window.configure(bg=PRIMARY_COLOR)
    
    tk.Label(result_window, text="Face Recognition Result", 
            font=(FONT_NAME, FONT_SIZE_HEADER, "bold"),
            bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)
    
    label = tk.Label(result_window, image=img_tk, bg=PRIMARY_COLOR)
    label.image = img_tk
    label.pack()
    
    tk.Button(result_window, text="Save Result", 
             command=lambda: save_image_result(display_image),
             font=(FONT_NAME, FONT_SIZE_BODY),
             bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

def face_recognition_video(known_encodings, known_names):
    cap = cv.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)  # Set a fixed resolution
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Only proceed if at least one face was found
        if face_locations:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare faces with known encodings
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                name = "Unknown"
                confidence = "0%"
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        confidence = f"{round((1 - face_distances[best_match_index]) * 100, 2)}%"
                
                # Draw rectangle around the face
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label with name and confidence below the face
                text = f"{name} {confidence}"
                cv.putText(frame, text, (left, bottom + 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting image
        cv.imshow('Face Recognition', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    cap.release()
    cv.destroyAllWindows()

def face_recognition_webcam(known_encodings, known_names):
    face_recognition_video(known_encodings, known_names)

def create_dual_option_window(title, image_func, video_func=None, webcam_func=None, encodings=None, names=None):
    def window_func():
        option_win = tk.Toplevel()
        option_win.title(title)
        option_win.iconbitmap(r'Assets/pattern.png.ico')
        option_win.geometry("600x500")
        option_win.resizable(False, False)
        option_win.configure(bg=PRIMARY_COLOR)

        tk.Label(option_win, text=title, font=(FONT_NAME, FONT_SIZE_TITLE, "bold"),
                bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=20)

        desc_text = "Select the input source for detection:"
        tk.Label(option_win, text=desc_text, font=(FONT_NAME, FONT_SIZE_BODY),
                bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

        button_frame = tk.Frame(option_win, bg=PRIMARY_COLOR)
        button_frame.pack(pady=20)

        if image_func:
            if encodings is not None and names is not None:
                tk.Button(button_frame, text="From Image",
                        command=lambda: image_func(encodings, names),
                        font=(FONT_NAME, FONT_SIZE_HEADER),
                        bg=SECONDARY_COLOR, fg=LIGHT_COLOR,
                        width=20, cursor='hand2').pack(pady=10)
            else:
                tk.Button(button_frame, text="From Image",
                        command=image_func,
                        font=(FONT_NAME, FONT_SIZE_HEADER),
                        bg=SECONDARY_COLOR, fg=LIGHT_COLOR,
                        width=20, cursor='hand2').pack(pady=10)

        if video_func:
            if encodings is not None and names is not None:
                tk.Button(button_frame, text="From Video",
                        command=lambda: video_func(encodings, names),
                        font=(FONT_NAME, FONT_SIZE_HEADER),
                        bg=SECONDARY_COLOR, fg=LIGHT_COLOR,
                        width=20, cursor='hand2').pack(pady=10)
            else:
                tk.Button(button_frame, text="From Video",
                        command=video_func,
                        font=(FONT_NAME, FONT_SIZE_HEADER),
                        bg=SECONDARY_COLOR, fg=LIGHT_COLOR,
                        width=20, cursor='hand2').pack(pady=10)

        if webcam_func:
            if encodings is not None and names is not None:
                tk.Button(button_frame, text="From Webcam",
                        command=lambda: webcam_func(encodings, names),
                        font=(FONT_NAME, FONT_SIZE_HEADER),
                        bg=SECONDARY_COLOR, fg=LIGHT_COLOR,
                        width=20, cursor='hand2').pack(pady=10)
            else:
                tk.Button(button_frame, text="From Webcam",
                        command=webcam_func,
                        font=(FONT_NAME, FONT_SIZE_HEADER),
                        bg=SECONDARY_COLOR, fg=LIGHT_COLOR,
                        width=20, cursor='hand2').pack(pady=10)

        tk.Button(option_win, text="Back", command=go_back,
                font=(FONT_NAME, FONT_SIZE_HEADER),
                bg=ACCENT_COLOR, fg=LIGHT_COLOR,
                width=15, cursor='hand2').pack(pady=20)
        
        tk.Button(option_win, text ='Home', command = go_back,
                  font=( FONT_NAME, FONT_SIZE_HEADER),
                   bg='green', fg=LIGHT_COLOR,
                   width=15, cursor='hand2').pack(pady=30)

        return option_win

    open_window(window_func)


def main_window():
    root = tk.Tk()
    root.title("Detection & Recognition System")
    root.iconbitmap(r'Assets/pattern.png.ico')
    root.geometry('800x700')
    root.resizable(False, False)

    try:
        bg_image = Image.open("Assets/pattern.jpg")
        bg_image = bg_image.resize((800, 700), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo
    except Exception as e:
        print(f"Error loading background image: {e}")
        root.configure(bg=PRIMARY_COLOR)

    main_frame = tk.Frame(root, bg=PRIMARY_COLOR, bd=2, relief=tk.RAISED)
    main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=600, height=500)

    encodings, names = load_known_faces('Images')

    tk.Label(main_frame, text="DETECTION SYSTEM", 
            font=(FONT_NAME, FONT_SIZE_TITLE, "bold"),
            bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=20)

    tk.Label(main_frame, 
            text="Advanced Computer Vision Applications",
            font=(FONT_NAME, FONT_SIZE_BODY),
            bg=PRIMARY_COLOR, fg=LIGHT_COLOR).pack(pady=5)

    menubar = tk.Menu(root)

    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Update Face Database", command=update_known_faces)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)

    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help", command=show_help)
    menubar.add_cascade(label="Help", menu=helpmenu)

    aboutmenu = tk.Menu(menubar, tearoff=0)
    aboutmenu.add_command(label="About", command=show_about)
    menubar.add_cascade(label="About", menu=aboutmenu)
    
    root.config(menu=menubar)

    button_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Object Detection",
             command=object_detection_webcam,
             font=(FONT_NAME, FONT_SIZE_HEADER),
             width=25,
             cursor='hand2',
             bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

    tk.Button(button_frame, text="License Plate Detection",
             command=lambda: create_dual_option_window(
                 "License Plate Detection",
                 license_plate_detection_image,
                 license_plate_detection_video),
                 font=(FONT_NAME, FONT_SIZE_HEADER),
                 width=25,
                 cursor='hand2',
                 bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

    tk.Button(button_frame, text="ANPR",
             command=lambda: create_dual_option_window(
                 "ANPR",
                 anpr_image,
                 anpr_video),
                 font=(FONT_NAME, FONT_SIZE_HEADER),
                 width=25,
                 cursor='hand2',
                 bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

    tk.Button(button_frame, text="Face Detection",
             command=lambda: create_dual_option_window(
                 "Face Detection",
                 face_detection_image,
                 None,
                 face_detection_webcam),
                 font=(FONT_NAME, FONT_SIZE_HEADER),
                 width=25,
                 cursor='hand2',
                 bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

    tk.Button(button_frame, text="Face Recognition",
              command=lambda: create_dual_option_window(
                  "Face Recognition",
                   face_recognition_image,
                   face_recognition_video,
                   None,
                   encodings,
                   names),
                   font=(FONT_NAME, FONT_SIZE_HEADER),
                   width=25,
                   cursor='hand2',
                   bg=SECONDARY_COLOR, fg=LIGHT_COLOR).pack(pady=10)

    footer_label = tk.Label(root,
                           text="© 2025 Detection & Recognition System | Computer Vision Project", 
                           font=(FONT_NAME, 8), 
                           bg=PRIMARY_COLOR, 
                           fg=LIGHT_COLOR)
    footer_label.pack(side=tk.BOTTOM, pady=10)
    
    window_stack.append(root)
    root.mainloop()

if __name__ == "__main__":
    if not os.path.exists("Assets"):
        os.makedirs("Assets")
    
    if not os.path.exists("Images"):
        os.makedirs("Images")
        messagebox.showinfo("Info", "Created 'Images' folder. Please add known face images here for recognition.")

    main_window()