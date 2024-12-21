import cv2
import time
import os
from datetime import datetime
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
from tts_player import play_gtts_text  # Import your TTS player function

# Base directory is the current working directory
BASE_DIR = os.getcwd()

# Directories for saving frames and logs
MOTION_FRAMES_DIR = os.path.join(BASE_DIR, "motion_frames_detected")
PERSON_FRAMES_DIR = os.path.join(BASE_DIR, "person_frames_detected")
MOTION_LOGS_DIR = os.path.join(BASE_DIR, "logs", "motion_logs")
PERSON_LOGS_DIR = os.path.join(BASE_DIR, "logs", "person_logs")

# Ensure the directories exist
os.makedirs(MOTION_FRAMES_DIR, exist_ok=True)
os.makedirs(PERSON_FRAMES_DIR, exist_ok=True)
os.makedirs(MOTION_LOGS_DIR, exist_ok=True)
os.makedirs(PERSON_LOGS_DIR, exist_ok=True)

# YOLO model for person detection
model = YOLO('yolov8n.pt')
allowed_labels = {"person", "cat", "dog"}  # Include pets

def format_detection_time(detection_time):
    """Format the detection time in a readable format."""
    hour = detection_time.strftime("%H")
    minute = detection_time.strftime("%M")
    second = detection_time.strftime("%S")
    return f"{int(hour):02d} horas, {int(minute):02d} minutos e {int(second):02d} segundos"

def save_frame(frame, folder, prefix):
    """Save a frame to the specified folder with a timestamped filename."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # High precision
        filename = os.path.join(folder, f"{prefix}_{timestamp}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"[{datetime.now()}] Frame saved: {filename}")
        else:
            print(f"[{datetime.now()}] Failed to save frame: {filename}")
    except Exception as e:
        print(f"[{datetime.now()}] Error saving frame: {e}")

def save_log(log_message, folder, prefix):
    """Save a log entry to a timestamped log file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(folder, f"{prefix}_{timestamp}.log")
        with open(log_filename, "w") as log_file:
            log_file.write(log_message + "\n")
        print(f"[{datetime.now()}] Log saved: {log_filename}")
    except Exception as e:
        print(f"[{datetime.now()}] Error saving log: {e}")

class WebcamApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Open video source (webcam)
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.video_source)

        # If possible, reduce or eliminate skipping frames for YOLO
        # to catch persons more quickly (e.g., set it to 1):
        self.frame_skip = 5  # Try setting this to 1 if you want YOLO every frame
        self.frame_count = 0
        
        # Separate save timers for motion and person detections
        self.last_saved_times = {"motion": 0, "person": 0}

        # Background subtractor for motion detection
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=5
        )
        self.motion_cooldown = time.time()
        self.motion_enabled = True
        self.motion_disabled_until = 0

        # List to store YOLO detections until the next YOLO run
        self.persistent_detections = []

        # Create a canvas to display the video feed
        self.canvas = Label(window)
        self.canvas.pack()

        # Button to quit
        self.btn_quit = Button(window, text="Quit", width=20, command=self.on_closing)
        self.btn_quit.pack(anchor="center", pady=10)

        # Start the main update loop
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if not ret:
            # Could not read frame
            self.window.after(10, self.update)
            return

        current_time = time.time()

        # ---------------------------------------------------------------------
        # 1. YOLO detection on every Nth frame (frame_skip)
        # ---------------------------------------------------------------------
        if self.frame_count % self.frame_skip == 0:
            results = model(frame)
            
            # Clear old detections only when we get fresh results
            self.persistent_detections = []

            for r in results:
                for box in r.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = model.names.get(cls_id, str(cls_id))

                    # Only keep detections for allowed labels
                    if label in allowed_labels:
                        self.persistent_detections.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max))
                        })

            # -----------------------------------------------------------------
            # 2. Person (or Pet) detection: TTS and disabling motion
            # -----------------------------------------------------------------
            if self.persistent_detections:
                detection_time = datetime.now()
                formatted_time = format_detection_time(detection_time)

                text_person = f"Pessoa detectada às {formatted_time}."
                text_animal = f"Animal detectado às {formatted_time}."

                for detection in self.persistent_detections:
                    if detection['label'] == 'person':
                        play_gtts_text(text_person, cooldown=5, speed=1.71)
                        # Disable motion detection for 20s if a person is found
                        self.motion_enabled = False
                        self.motion_disabled_until = current_time + 20
                        break
                    # elif detection['label'] in {"cat", "dog"}:
                    #     play_gtts_text(text_animal, cooldown=5, speed=1.71)
                    #     break



        # ---------------------------------------------------------------------
        # 4. Motion detection (only if motion is enabled)
        # ---------------------------------------------------------------------
        if self.motion_enabled:
            motion_mask = self.motion_detector.apply(frame)
            _, motion_thresh = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)
            motion_contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check for significant motion in the frame
            significant_motion = any(cv2.contourArea(contour) > 500 for contour in motion_contours)

            # If there's motion and the cooldown for saving frames is over:
            if significant_motion:
                save_frame(frame, MOTION_FRAMES_DIR, "motion")
                self.last_saved_times["motion"] = current_time
                motion_time = datetime.now()
                formatted_motion_time = format_detection_time(motion_time)
                motion_text = f"Movimento detectado às {formatted_motion_time}."
                play_gtts_text(motion_text, cooldown=10, speed=1.61)

        else:
            # If motion was disabled due to a recent person detection, check if the cooldown has expired.
            if current_time >= self.motion_disabled_until:
                self.motion_enabled = True

        # ---------------------------------------------------------------------
        # 5. Always draw the last YOLO detections (to avoid blinking)
        # ---------------------------------------------------------------------
        for detection in self.persistent_detections:
            x_min, y_min, x_max, y_max = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']

            color = (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame,
                        f"{label} {confidence:.2f}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)
                        # -----------------------------------------------------------------
            # 3. Save frames if a person or pet is detected
            # -----------------------------------------------------------------
            # The "0.2" threshold means 5 frames a second at 30 FPS
            if self.persistent_detections and (current_time - self.last_saved_times["person"] >= 0.2):
                save_frame(frame, PERSON_FRAMES_DIR, "person_pet")
                self.last_saved_times["person"] = current_time

        # ---------------------------------------------------------------------
        # 6. Convert to RGB and display in Tkinter
        # ---------------------------------------------------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        self.frame_count += 1

        # Schedule the next update
        self.window.after(10, self.update)

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

# Run the application
if __name__ == "__main__":
    root = Tk()
    WebcamApp(root, "Webcam Motion Detection")
