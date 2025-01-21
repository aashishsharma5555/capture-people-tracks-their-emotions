import cv2
import tkinter as tk
from deepface import DeepFace
from PIL import Image, ImageTk  #use PIL to convert OpenCV images for Tkinter

# Load the face  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# live_capture
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video.isOpened():
    raise IOError("Cannot open webcam")

# Flag to detect emotion or not
detect_emotion = False
running = True

# Tk window setup
root = tk.Tk()
root.title("Emotion Detection")

# get keyword useing screen resolution (width, height)
screen_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# a canvas to display the OpenCV video feed
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

# Buttons
def close_camera():
    global running
    running = False
    video.release()
    root.quit()  # Close window

def start_emotion_detection():
    global detect_emotion
    detect_emotion = True
    emotion_button.config(text="Turn Off Emotion Detection", command=stop_emotion_detection)

def stop_emotion_detection():
    global detect_emotion
    detect_emotion = False
    emotion_button.config(text="Turn On Emotion Detection", command=start_emotion_detection)

# control buttons
close_button = tk.Button(root, text="Close Camera", command=close_camera)
close_button.pack(side="left", padx=10, pady=10)

emotion_button = tk.Button(root, text="Turn On Emotion Detection", command=start_emotion_detection)
emotion_button.pack(side="right", padx=10, pady=10)

# camera feed loop with emotion detection
def video_loop():
    global running, detect_emotion
    
    # Capture frame from the camera
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #the number of people detected
    num_people = 0

    for x, y, w, h in faces:
        # rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (89, 2, 236), 1)
        
        # Detect smile with the detected face
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest (the face)
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        if len(smiles) > 0:
            # If smile is detected, add text indicating smile
            cv2.putText(frame, "Smile Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Emotion detection if the button is clicked
        if detect_emotion:
            try:
                # Analyze the emotions using DeepFace
                analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analyze[0]['dominant_emotion']
                
                # Display the dominant emotion
                cv2.putText(frame, dominant_emotion, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (224, 77, 176), 2)
                
                # Display the full set of emotions and their confidence levels
                emotions = analyze[0]['emotions']
                y_offset = y + h + 40
                for emotion, score in emotions.items():
                    cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    y_offset += 20
            except Exception as e:
                print('Emotion detection failed:', e)
        
        num_people += 1  # Increase the count of detected people

    # Display the number of people on the left side
    cv2.putText(frame, f"People Count: {num_people}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to a format Tkinter understands using PIL
    img = Image.fromarray(frame_rgb)  # Convert to Image
    img_tk = ImageTk.PhotoImage(image=img)  # Convert to Tkinter format

    # Update the canvas with image
    canvas.create_image(0, 0, image=img_tk, anchor="nw")

    # Keep reference to image to avoid garbage collection
    canvas.image = img_tk

    if running:
        # Repeat process
        canvas.after(10, video_loop)

# Startvideo loop
video_loop()

# Run Tk main loop
root.mainloop()

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
