"""
Eye Tracker Module - Core classes for eye detection and alarm functionality.
Uses MediaPipe for eye detection and blink/closure detection.
"""

import cv2
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def get_base_dir() -> Path:
    """Get base directory, handling PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return Path(sys._MEIPASS)
    return Path(__file__).parent


BASE_DIR = get_base_dir()
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "threshold": 3.0,
    "interval": 0.5,
    "show_preview": True,
    "ear_threshold": 0.2,
    "paths": {
        "models_dir": "models",
        "model_file": "face_landmarker.task",
        "alarm_video": "alarm/skeleton banging shield.mp4"
    },
    "urls": {
        "model_download": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    }
}


def load_config() -> dict:
    """Load configuration from config.json, creating it with defaults if missing."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    else:
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG.copy()


def get_paths(config: dict) -> dict:
    """Resolve paths from config relative to base directory."""
    paths = config.get("paths", DEFAULT_CONFIG["paths"])
    models_dir = BASE_DIR / paths["models_dir"]
    return {
        "models_dir": models_dir,
        "model_file": models_dir / paths["model_file"],
        "alarm_video": BASE_DIR / paths["alarm_video"]
    }


def download_model(config: dict):
    """Download the MediaPipe face landmarker model if not present."""
    paths = get_paths(config)
    model_path = paths["model_file"]

    if model_path.exists():
        return

    print("Downloading MediaPipe face landmarker model...")
    paths["models_dir"].mkdir(exist_ok=True)
    model_url = config.get("urls", DEFAULT_CONFIG["urls"])["model_download"]
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded.")


class EyeDetector:
    """MediaPipe-based eye detector with open/closed detection."""

    # MediaPipe Face Landmarker eye landmark indices
    # Left eye (from user's perspective, right side of image)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Right eye (from user's perspective, left side of image)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def __init__(self, config: dict, ear_threshold: float = 0.2):
        """
        Initialize the eye detector.

        Args:
            config: Configuration dictionary
            ear_threshold: Eye Aspect Ratio threshold below which eyes are considered closed
        """
        self.ear_threshold = ear_threshold

        download_model(config)
        paths = get_paths(config)

        base_options = python.BaseOptions(model_asset_path=str(paths["model_file"]))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def calculate_ear(self, landmarks, eye_indices, img_w, img_h) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for an eye.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Higher EAR = eye open, Lower EAR = eye closed
        """
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            points.append([lm.x * img_w, lm.y * img_h])

        points = np.array(points)

        # Vertical distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])

        # Horizontal distance
        h = np.linalg.norm(points[0] - points[3])

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect(self, frame) -> dict:
        """
        Detect eyes and determine if they are open or closed.

        Returns:
            dict with keys:
                - face_detected: bool
                - eyes_open: bool
                - left_ear: float (Eye Aspect Ratio)
                - right_ear: float
                - landmarks: list of eye landmarks for drawing
        """
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return {
                "face_detected": False,
                "eyes_open": False,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "landmarks": None
            }

        face_landmarks = results.face_landmarks[0]

        left_ear = self.calculate_ear(face_landmarks, self.LEFT_EYE, img_w, img_h)
        right_ear = self.calculate_ear(face_landmarks, self.RIGHT_EYE, img_w, img_h)

        avg_ear = (left_ear + right_ear) / 2.0
        eyes_open = avg_ear >= self.ear_threshold

        # Get landmark positions for drawing
        left_eye_points = []
        right_eye_points = []

        for idx in self.LEFT_EYE:
            lm = face_landmarks[idx]
            left_eye_points.append((int(lm.x * img_w), int(lm.y * img_h)))

        for idx in self.RIGHT_EYE:
            lm = face_landmarks[idx]
            right_eye_points.append((int(lm.x * img_w), int(lm.y * img_h)))

        return {
            "face_detected": True,
            "eyes_open": eyes_open,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "landmarks": {
                "left_eye": left_eye_points,
                "right_eye": right_eye_points
            }
        }


class AlarmPlayer:
    """Plays alarm video with audio on loop."""

    # Common ffplay locations on macOS
    FFPLAY_PATHS = [
        "/opt/homebrew/bin/ffplay",  # Apple Silicon homebrew
        "/usr/local/bin/ffplay",      # Intel homebrew
        "ffplay"                       # System PATH
    ]

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.process = None
        self.is_playing = False
        self.ffplay_path = self._find_ffplay()

    def _find_ffplay(self) -> str | None:
        """Find ffplay executable."""
        for path in self.FFPLAY_PATHS:
            try:
                result = subprocess.run(
                    [path, "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if result.returncode == 0:
                    return path
            except FileNotFoundError:
                continue
        return None

    def start(self):
        """Start playing the alarm video with audio."""
        if self.is_playing:
            return

        if not self.video_path.exists():
            print(f"Warning: Alarm video not found at {self.video_path}")
            return

        if self.ffplay_path:
            try:
                self.process = subprocess.Popen(
                    [
                        self.ffplay_path,
                        "-loop", "0",
                        "-window_title", "WAKE UP!",
                        str(self.video_path)
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.is_playing = True
                return
            except Exception as e:
                print(f"Warning: ffplay failed: {e}")

        # Fallback: Use QuickTime with AppleScript for looping
        print("Using QuickTime Player fallback...")
        self._start_quicktime()

    def _start_quicktime(self):
        """Start video with QuickTime Player using AppleScript."""
        script = f'''
        tell application "QuickTime Player"
            activate
            open POSIX file "{self.video_path}"
            delay 0.5
            tell document 1
                set looping to true
                play
            end tell
        end tell
        '''
        try:
            subprocess.Popen(["osascript", "-e", script],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            self.is_playing = True
        except Exception as e:
            print(f"Warning: QuickTime fallback failed: {e}")

    def stop(self):
        """Stop playing the alarm video."""
        if not self.is_playing:
            return

        self.is_playing = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        else:
            # Stop QuickTime if it was used
            script = '''
            tell application "QuickTime Player"
                if (count of documents) > 0 then
                    close document 1
                end if
            end tell
            '''
            subprocess.run(["osascript", "-e", script],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

    def update(self):
        """Check if player is still running."""
        if self.is_playing and self.process:
            if self.process.poll() is not None:
                self.is_playing = False
                self.process = None


class EyeTracker:
    def __init__(self, config: dict):
        """
        Initialize the eye tracker.

        Args:
            config: Configuration dictionary
        """
        self.check_interval = config.get("interval", 0.5)
        self.missing_threshold = config.get("threshold", 3.0)
        self.last_eyes_open = time.time()
        self.notification_sent = False

        ear_threshold = config.get("ear_threshold", 0.2)
        paths = get_paths(config)

        self.detector = EyeDetector(config, ear_threshold=ear_threshold)
        self.alarm = AlarmPlayer(paths["alarm_video"])

    def send_notification(self, title: str, message: str):
        """Send a macOS notification using osascript."""
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True)

    def draw_detections(self, frame, detection: dict):
        """Draw eye landmarks and status on frame."""
        if detection["landmarks"]:
            # Draw left eye
            for point in detection["landmarks"]["left_eye"]:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Draw right eye
            for point in detection["landmarks"]["right_eye"]:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Draw EAR values
            cv2.putText(frame, f"EAR: {detection['avg_ear']:.2f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def run(self, show_preview: bool = True):
        """
        Main loop - continuously monitors camera for eyes.

        Args:
            show_preview: Whether to show a preview window with detections
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Eye tracker started (MediaPipe). Press 'q' to quit.")
        print(f"Notification will trigger after {self.missing_threshold}s with eyes closed.")
        print(f"EAR threshold: {self.detector.ear_threshold}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                current_time = time.time()
                detection = self.detector.detect(frame)

                eyes_open = detection["face_detected"] and detection["eyes_open"]

                if eyes_open:
                    self.last_eyes_open = current_time
                    self.notification_sent = False
                    self.alarm.stop()
                else:
                    time_since_open = current_time - self.last_eyes_open

                    if time_since_open >= self.missing_threshold and not self.notification_sent:
                        self.send_notification(
                            "Eyes Closed!",
                            "Wake up! Your eyes have been closed."
                        )
                        self.alarm.start()
                        self.notification_sent = True
                        print(f"[{time.strftime('%H:%M:%S')}] Alarm triggered - eyes closed")

                # Update alarm video if playing
                self.alarm.update()

                if show_preview:
                    frame = self.draw_detections(frame, detection)

                    if not detection["face_detected"]:
                        status = "No face detected"
                        color = (0, 165, 255)  # Orange
                    elif detection["eyes_open"]:
                        status = "Eyes: OPEN"
                        color = (0, 255, 0)  # Green
                    else:
                        status = "Eyes: CLOSED"
                        color = (0, 0, 255)  # Red

                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               1, color, 2)

                    cv2.imshow("Eye Tracker", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(self.check_interval)

        finally:
            self.alarm.stop()
            cap.release()
            cv2.destroyAllWindows()
            print("Eye tracker stopped.")


