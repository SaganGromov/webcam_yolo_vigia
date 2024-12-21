import cv2
import time
import os
from datetime import datetime
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

class HeadlessMotionDetector:
    def __init__(self, video_source=0):
        """Initialize the headless motion detection system."""
        # Open video source (webcam or file)
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # If your machine can handle it, you can set frame_skip=1 to run YOLO every frame
        self.frame_skip = 5
        self.frame_count = 0

        # Track last saved times to avoid saving too many frames
        self.last_saved_times = {
            "motion": 0,
            "person": 0
        }

        # Create background subtractor for motion detection
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=10
        )

        self.motion_enabled = True
        self.motion_disabled_until = 0

        # List for storing YOLO detections between frames
        self.persistent_detections = []

    def run(self):
        """Run the main detection loop until interrupted."""
        print("[INFO] Starting headless motion/person detection. Press Ctrl+C to stop.")
        try:
            while True:
                ret, frame = self.vid.read()
                if not ret:
                    print("[ERROR] Failed to read frame from source.")
                    break

                current_time = time.time()

                # -----------------------------
                # 1. YOLO detection on Nth frame
                # -----------------------------
                if self.frame_count % self.frame_skip == 0:
                    results = model(frame)
                    
                    # Clear old detections, then populate with current YOLO results
                    self.persistent_detections = []

                    for r in results:
                        for box in r.boxes:
                            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            label = model.names.get(cls_id, str(cls_id))

                            if label in allowed_labels:
                                self.persistent_detections.append({
                                    'label': label,
                                    'confidence': confidence,
                                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max))
                                })

                    # -----------------------------
                    # 2. TTS + motion disabling if a person is detected
                    # -----------------------------
                    if self.persistent_detections:
                        detection_time = datetime.now()
                        formatted_time = format_detection_time(detection_time)
                        text_person = f"Pessoa detectada às {formatted_time}."
                        text_animal = f"Animal detectado às {formatted_time}."

                        for detection in self.persistent_detections:
                            if detection['label'] == 'person':
                                play_gtts_text(text_person, cooldown=3, speed=1.71)
                                self.motion_enabled = False
                                self.motion_disabled_until = current_time + 20
                                break
                            elif detection['label'] in {'cat', 'dog'}:
                                play_gtts_text(text_animal, cooldown=3, speed=1.71)
                                break

                # -----------------------------
                # 3. Draw bounding boxes
                # -----------------------------
                if self.persistent_detections:
                    # Draw bounding boxes directly on `frame`
                    for detection in self.persistent_detections:
                        x_min, y_min, x_max, y_max = detection['bbox']
                        label = detection['label']
                        confidence = detection['confidence']
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(
                            frame,
                            f"{label} {confidence:.2f}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

                # -----------------------------
                # 4. Save frames (person/pet)
                # -----------------------------
                # Do it after drawing boxes, so the image on disk shows annotations.
                if self.persistent_detections and (current_time - self.last_saved_times["person"] >= 0.2):
                    save_frame(frame, PERSON_FRAMES_DIR, "person_pet")
                    self.last_saved_times["person"] = current_time

                # -----------------------------
                # 5. Motion detection (if enabled)
                # -----------------------------
                if self.motion_enabled:
                    motion_mask = self.motion_detector.apply(frame)
                    _, motion_thresh = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)
                    motion_contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Check for significant motion
                    significant_motion = any(cv2.contourArea(c) > 500 for c in motion_contours)
                    
                    if significant_motion and (current_time - self.last_saved_times["motion"] >= 0.2):
                        save_frame(frame, MOTION_FRAMES_DIR, "motion")
                        self.last_saved_times["motion"] = current_time
                        motion_time = datetime.now()
                        formatted_motion_time = format_detection_time(motion_time)
                        motion_text = f"Movimento detectado!"
                        play_gtts_text(motion_text, cooldown=10, speed=2.22)
                else:
                    # Check if we can re-enable motion detection
                    if current_time >= self.motion_disabled_until:
                        self.motion_enabled = True

                self.frame_count += 1

                # Small delay to prevent 100% CPU usage
                # (not strictly necessary, but can help smooth performance)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("[INFO] Stopping due to Ctrl + C.")
        finally:
            self.vid.release()
            print("[INFO] Video source released.")

# -------------------------------
# MAIN ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    detector = HeadlessMotionDetector(video_source=0)
    detector.run()

