"""
Driver Monitoring System Module for Q-DRIVE Cortex
Author: Arjun Joshi
Date: 10.14.2025
Description: Handles eye tracking, gaze estimation, drowsiness detection, and distraction monitoring
Cross-platform compatilbility with both CARLA simulator and Orin field deployment - optimization for edge devices.

"""

import os
import sys
import subprocess
import logging

# ============================
# GPU ALLOCATION - MUST BE SET BEFORE ANY GPU LIBRARY IMPORTS
# ============================

gpu_id = 0
gpu_available = False
actual_device_id = 0


def prompt_gpu_id() -> int:
    """Prompt user for GPU selection, defaulting to GPU 1 if available, else GPU 0"""
    gpu_choice = 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_list = []
        print("\nAvailable GPUs:")
        for line in result.stdout.strip().split("\n"):
            if line:
                idx, name = line.split(", ", 1)
                gpu_list.append((idx.strip(), name.strip()))
                print(f"  {idx.strip()}) {name.strip()}")

        # Establish default gpu as 1 if it exists, otherwise 0
        if len(gpu_list) > 1:
            default_gpu = 1
            user_input = input(
                f"\nSelect GPU ID for DMS (default {default_gpu}): "
            ).strip()
            gpu_choice = int(user_input) if user_input else default_gpu
        else:
            default_gpu = 0
            user_input = input(
                f"\nSelect GPU ID for DMS (default {default_gpu}): "
            ).strip()
            gpu_choice = int(user_input) if user_input else default_gpu

        return gpu_choice
    except Exception as e:
        logging.warning(f"Could not query GPUs: {e}")
        print("GPU detection failed, will attempt to use primary GPU (0)")
        return 0


# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing any GPU libraries
try:
    gpu_id = prompt_gpu_id()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    # Suppress TensorFlow/protobuf warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning logs

    gpu_available = True
    # After setting CUDA_VISIBLE_DEVICES, the selected GPU becomes device 0 in PyTorch/TF
    # But we keep track of the original GPU ID for display purposes
    actual_device_id = 0  # This is what PyTorch/TF will see
    selected_gpu_id = gpu_id  # This is what we selected (for display)
    logging.info(f"Physical GPU {gpu_id} selected for DMS (mapped to logical device 0 in frameworks)")
    print(f"GPU Configuration: Using Physical GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
except Exception as e:
    logging.warning(f"GPU setup failed, will use CPU: {e}")
    gpu_available = False
    actual_device_id = 0
    selected_gpu_id = -1  # -1 indicates CPU mode

# NOW it's safe to import GPU libraries (they will only see CUDA_VISIBLE_DEVICES)
import cv2 as cv
import numpy as np
import pandas as pd
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from collections import deque

# Suppress protobuf deprecation warnings (compatibility issue between protobuf 6.x and MediaPipe 0.10.x)
import warnings
from io import StringIO

warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='.*GetPrototype.*')

# Capture and suppress stderr during MediaPipe import (protobuf compatibility issue)
# This suppresses the "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" message
_original_stderr = sys.stderr
sys.stderr = StringIO()

import mediapipe as mp

# Restore stderr
sys.stderr = _original_stderr

import torch, torchvision

# MediaPipe landmark indices
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Face orientation landmarks for head pose
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


class AlertLevel(Enum):
    """Alert severity levels for driver state"""

    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class EyeMetrics:
    """Per-eye tracking metrics"""

    center: Tuple[float, float] = (0, 0)
    radius: float = 0.0
    aspect_ratio: float = 1.0  # For blink detection
    is_closed: bool = False
    iris_position: Tuple[float, float] = (0, 0)  # Normalized (-1 to 1)


@dataclass
class DriverState:
    """Comprehensive driver state for scoring and alerts"""

    timestamp: float = 0.0

    # Detection status
    face_detected: bool = False
    eyes_detected: bool = False

    # Eye metrics
    left_eye: EyeMetrics = field(default_factory=EyeMetrics)
    right_eye: EyeMetrics = field(default_factory=EyeMetrics)

    # Aggregate metrics
    eyes_closed_duration_ms: float = 0.0
    blink_rate: float = 0.0  # blinks per minute
    gaze_vector: Tuple[float, float, float] = (0, 0, 0)  # Default to null
    head_pose: Tuple[float, float, float] = (0, 0, 0)  # yaw, pitch, roll

    # Alert states (None when no detection)
    drowsiness_score: Optional[float] = None  # 0-1
    distraction_score: Optional[float] = None  # 0-1
    attention_score: Optional[float] = None  # 0-1 (inverse of distraction)

    # Event flags
    microsleep_detected: bool = False
    looking_at_phone: bool = False
    head_down_event: bool = False

    alert_level: AlertLevel = AlertLevel.NORMAL


class DMS:
    """
    Main Driver Monitoring System class
    Processes webcam/camera feed to monitor driver state
    Thread-safe for integration with controls_queue pattern
    """

    def __init__(self, camera_index=0, use_tensorrt=False):
        """
        Initialize DMS with specified camera
        Args:
            camera_index: OpenCV camera index or path (for Logitech C615, usually 0)
            use_tensorrt: Enable TensorRT optimization for Orin deployment
        """
        self.camera_index = camera_index
        self.use_tensorrt = use_tensorrt
        log_df = pd.DataFrame()
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Model information for debug logging
        self.model_name = "MediaPipe Face Mesh"
        self.model_version = "v1.0"
        self.backend = "TensorRT" if use_tensorrt else "TensorFlow Lite"
        logging.info(f"[MODEL] Using {self.model_name} ({self.model_version}) with {self.backend} backend")

        # Thread management
        self._running = False
        self._capture_thread = None
        self._process_thread = None

        # Producer-consumer queues (matching controls_queue pattern)
        self.frame_queue = queue.Queue(maxsize=2)  # Latest frames only
        self.state_queue = queue.Queue(maxsize=10)  # Processed states

        # State tracking
        self.current_state = DriverState()
        self._state_history = deque(maxlen=150)  # 5 seconds at 30fps

        # Calibration and thresholds
        self.calibration = {
            "eye_closed_threshold": 0.2,  # Eye aspect ratio
            "microsleep_ms": 500,
            "distraction_angle_deg": 30,
            "phone_zone": (0.3, 0.7, 0.7, 0.9),  # Normalized bbox for phone area
            "calibrated": False,
            "neutral_gaze": (0, 0, 1),  # Will be set during calibration
            "neutral_head_pose": (0, 0, 0),  # Will be set during calibration
        }

        # Timing
        self._last_blink_time = 0
        self._eye_closed_start = 0
        self._blink_times = deque(maxlen=20)

        # Performance metrics
        self.fps = 0
        self._frame_times = deque(maxlen=30)

        # Output timing control
        self._last_output_time = 0
        self.output_interval = 0.5  # seconds

        logging.info(f"DMS initialized with camera {camera_index}")

    def start(self):
        """Start DMS processing threads"""
        if self._running:
            return

        self._running = True

        # Start capture thread (producer)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="DMS_Capture"
        )
        self._capture_thread.daemon = True
        self._capture_thread.start()

        # Start processing thread (consumer)
        self._process_thread = threading.Thread(
            target=self._process_loop, name="DMS_Process"
        )
        self._process_thread.daemon = True
        self._process_thread.start()

        logging.info("DMS threads started")

    def stop(self):
        """Stop DMS processing"""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
        logging.info("DMS stopped")

    def _capture_loop(self):
        """Camera capture thread - produces frames"""
        cap = cv.VideoCapture(self.camera_index)

        # Configure for Logitech C615
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FPS, 30)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logging.warning("DMS: Failed to capture frame")
                time.sleep(0.1)
                continue

            # Flip for mirror effect (driver facing camera)
            frame = cv.flip(frame, 1)

            # Non-blocking put
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Drop oldest frame if queue full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except:
                    pass

        cap.release()

    def _process_loop(self):
        """Processing thread - consumes frames and produces driver states"""
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Process frame
            state = self._process_frame(frame)

            # Update state history
            self._state_history.append(state)
            self.current_state = state

            # Calculate aggregate metrics
            self._update_aggregate_metrics(state)

            # Publish state
            try:
                self.state_queue.put(state, block=False)
            except queue.Full:
                # Drop oldest state
                try:
                    self.state_queue.get_nowait()
                    self.state_queue.put(state, block=False)
                except:
                    pass

            # Update FPS
            self._frame_times.append(time.time())
            if len(self._frame_times) > 1:
                self.fps = len(self._frame_times) / (
                    self._frame_times[-1] - self._frame_times[0]
                )

    def _process_frame(self, frame) -> DriverState:
        """
        Process single frame to extract driver state
        This is where your MediaPipe code integrates
        """
        state = DriverState(timestamp=time.time())

        # Debug: Log model being used (every 0.5 seconds)
        current_time = time.time()
        if current_time - self._last_output_time >= self.output_interval:
            logging.debug(f"[MODEL-INFERENCE] Processing frame with {self.model_name} on {self.backend}")
            self._last_output_time = current_time

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in landmarks.landmark
                ]
            )

            # Mark detection status
            state.face_detected = True
            state.eyes_detected = True

            # Process eyes
            state.left_eye = self._process_eye(mesh_points, LEFT_EYE, LEFT_IRIS)
            state.right_eye = self._process_eye(mesh_points, RIGHT_EYE, RIGHT_IRIS)

            # Calculate head pose
            state.head_pose = self._estimate_head_pose(landmarks, img_w, img_h)

            # Calculate gaze vector (simplified for now)
            state.gaze_vector = self._estimate_gaze(
                state.left_eye, state.right_eye, state.head_pose
            )

            # Detect blinks and microsleeps
            self._detect_blinks(state)

            # Calculate attention metrics
            state.distraction_score = self._calculate_distraction(state)
            state.drowsiness_score = self._calculate_drowsiness(state)
            state.attention_score = 1.0 - max(
                state.distraction_score, state.drowsiness_score
            )

            # Set alert level
            state.alert_level = self._determine_alert_level(state)
        else:
            # No face detected - set critical alert
            state.face_detected = False
            state.eyes_detected = False
            state.drowsiness_score = 0.0
            state.distraction_score = 0.0
            state.attention_score = 0.0
            state.alert_level = AlertLevel.CRITICAL

        return state

    def _process_eye(self, mesh_points, eye_indices, iris_indices) -> EyeMetrics:
        """Process individual eye metrics"""
        eye = EyeMetrics()

        # Get iris center and radius
        (cx, cy), radius = cv.minEnclosingCircle(mesh_points[iris_indices])
        eye.center = (cx, cy)
        eye.radius = radius

        # Calculate eye aspect ratio for blink detection
        eye_points = mesh_points[eye_indices]
        eye_height = np.linalg.norm(eye_points[1] - eye_points[5])  # Simplified
        eye_width = np.linalg.norm(eye_points[0] - eye_points[8])

        if eye_width > 0:
            eye.aspect_ratio = eye_height / eye_width

        # Check if eye is closed
        eye.is_closed = eye.aspect_ratio < self.calibration["eye_closed_threshold"]

        # Calculate normalized iris position within eye
        if len(eye_points) > 0:
            eye_center = np.mean(eye_points, axis=0)
            eye.iris_position = (
                (cx - eye_center[0]) / (eye_width / 2) if eye_width > 0 else 0,
                (cy - eye_center[1]) / (eye_height / 2) if eye_height > 0 else 0,
            )

        return eye

    def _estimate_head_pose(
        self, landmarks, img_w, img_h
    ) -> Tuple[float, float, float]:
        """Estimate head pose (yaw, pitch, roll) from facial landmarks"""
        # Simplified PnP-based pose estimation
        # In production, use proper 3D model points and camera calibration

        # Key facial points for pose
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[152]
        left_eye_outer = landmarks.landmark[33]
        right_eye_outer = landmarks.landmark[133]

        # Simple approximation (replace with proper PnP solve)
        yaw = (nose_tip.x - 0.5) * 60  # -30 to +30 degrees
        pitch = (nose_tip.y - 0.5) * 40  # -20 to +20 degrees
        roll = 0  # Would need more complex calculation

        return (yaw, pitch, roll)

    def _estimate_gaze(
        self,
        left_eye: EyeMetrics,
        right_eye: EyeMetrics,
        head_pose: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Estimate 3D gaze vector from eye and head pose"""
        # Combine iris positions with head pose
        avg_iris_x = (left_eye.iris_position[0] + right_eye.iris_position[0]) / 2
        avg_iris_y = (left_eye.iris_position[1] + right_eye.iris_position[1]) / 2

        # Add iris deviation to head pose
        gaze_yaw = head_pose[0] + avg_iris_x * 15  # degrees
        gaze_pitch = head_pose[1] + avg_iris_y * 10

        # Convert to unit vector
        yaw_rad = np.radians(gaze_yaw)
        pitch_rad = np.radians(gaze_pitch)

        x = np.sin(yaw_rad) * np.cos(pitch_rad)
        y = np.sin(pitch_rad)
        z = np.cos(yaw_rad) * np.cos(pitch_rad)

        return (x, y, z)

    def _detect_blinks(self, state: DriverState):
        """Detect blinks and microsleeps"""
        both_closed = state.left_eye.is_closed and state.right_eye.is_closed

        if both_closed:
            if self._eye_closed_start == 0:
                self._eye_closed_start = time.time()

            state.eyes_closed_duration_ms = (
                time.time() - self._eye_closed_start
            ) * 1000

            # Detect microsleep
            if state.eyes_closed_duration_ms > self.calibration["microsleep_ms"]:
                state.microsleep_detected = True
        else:
            if self._eye_closed_start > 0:
                # Blink completed
                blink_duration = time.time() - self._eye_closed_start
                if 0.1 < blink_duration < 0.4:  # Normal blink duration
                    self._blink_times.append(time.time())

                self._eye_closed_start = 0

            state.eyes_closed_duration_ms = 0

        # Calculate blink rate
        if len(self._blink_times) > 1:
            time_span = self._blink_times[-1] - self._blink_times[0]
            if time_span > 0:
                state.blink_rate = (len(self._blink_times) / time_span) * 60

    def _calculate_distraction(self, state: DriverState) -> float:
        """Calculate distraction score based on gaze and head pose"""
        score = 0.0

        # Check if looking away from road (forward is z=1)
        gaze_forward = state.gaze_vector[2]
        if gaze_forward < 0.8:  # Not looking forward
            score += 0.3

        # Check head pose deviation
        yaw_abs = abs(state.head_pose[0])
        if yaw_abs > self.calibration["distraction_angle_deg"]:
            score += 0.4

        # Check for phone zone gaze
        if self._is_looking_at_phone_zone(state):
            score += 0.3
            state.looking_at_phone = True

        return min(score, 1.0)

    def _calculate_drowsiness(self, state: DriverState) -> float:
        """Calculate drowsiness score"""
        score = 0.0

        # Long eye closure
        if state.eyes_closed_duration_ms > 200:
            score += min(state.eyes_closed_duration_ms / 1000, 0.5)

        # Microsleep detection
        if state.microsleep_detected:
            score = 1.0

        # Abnormal blink rate
        if state.blink_rate > 30 or state.blink_rate < 10:
            score += 0.2

        # Head nodding (pitch deviation)
        if state.head_pose[1] < -15:  # Head tilted down
            score += 0.3
            state.head_down_event = True

        return min(score, 1.0)

    def _is_looking_at_phone_zone(self, state: DriverState) -> bool:
        """Check if driver is looking at typical phone position"""
        # Phone typically held lower and to the right
        return (
            state.gaze_vector[1] < -0.3  # Looking down
            and abs(state.gaze_vector[0]) > 0.3
        )  # Looking to side

    def _determine_alert_level(self, state: DriverState) -> AlertLevel:
        """Determine overall alert level"""
        max_score = max(state.distraction_score, state.drowsiness_score)

        if max_score < 0.3:
            return AlertLevel.NORMAL
        elif max_score < 0.5:
            return AlertLevel.CAUTION
        elif max_score < 0.7:
            return AlertLevel.WARNING
        else:
            return AlertLevel.CRITICAL

    def _update_aggregate_metrics(self, state: DriverState):
        """Update running metrics from state history"""
        if len(self._state_history) < 2:
            return

        # Additional processing for trends
        recent_states = list(self._state_history)[-30:]  # Last second

        # Check for patterns (e.g., repeated microsleeps)
        microsleep_count = sum(1 for s in recent_states if s.microsleep_detected)
        if microsleep_count > 2 and state.drowsiness_score is not None:
            state.drowsiness_score = min(state.drowsiness_score + 0.2, 1.0)

    def calibrate(self, duration_seconds=5):
        """
        Calibrate the system by capturing neutral driver position
        Args:
            duration_seconds: How long to sample for baseline (default 5 seconds)
        """
        print("\n" + "="*100)
        print("CALIBRATION MODE")
        print("="*100)
        print("\nPlease sit in a comfortable driving position and look straight ahead.")
        print(f"Calibration will run for {duration_seconds} seconds...")
        print("Starting in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("GO! Stay still and look at the road ahead.\n")

        gaze_samples = []
        head_pose_samples = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            state = self.get_latest_state()
            if state and state.face_detected:
                gaze_samples.append(state.gaze_vector)
                head_pose_samples.append(state.head_pose)
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                print(f"\rCalibrating... {remaining:.1f}s remaining", end="", flush=True)
            time.sleep(0.1)

        print("\n\nCalibration complete!")

        if len(gaze_samples) > 10:
            # Calculate average neutral position
            avg_gaze = tuple(np.mean(gaze_samples, axis=0))
            avg_head_pose = tuple(np.mean(head_pose_samples, axis=0))

            self.calibration["neutral_gaze"] = avg_gaze
            self.calibration["neutral_head_pose"] = avg_head_pose
            self.calibration["calibrated"] = True

            print(f"\nNeutral gaze calibrated to: {avg_gaze}")
            print(f"Neutral head pose calibrated to: {avg_head_pose}")
            print(f"Samples collected: {len(gaze_samples)}")
        else:
            print("\nERROR: Not enough face detection samples. Please try again.")
            self.calibration["calibrated"] = False

        print(f"\n{'='*100}\n")
        time.sleep(2)

    def get_latest_state(self) -> Optional[DriverState]:
        """Get most recent driver state (non-blocking)"""
        try:
            # Drain queue to get latest
            state = None
            while True:
                state = self.state_queue.get_nowait()
        except queue.Empty:
            return state if state else self.current_state

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Get annotated frame for debugging/visualization"""
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return None

        # Draw annotations
        if self.current_state:
            self._draw_debug_overlay(frame, self.current_state)

        return frame

    def _draw_debug_overlay(self, frame, state: DriverState):
        """Draw debug information on frame"""
        h, w = frame.shape[:2]

        # Draw eye centers and iris
        for eye, label in [(state.left_eye, "L"), (state.right_eye, "R")]:
            if eye.center[0] > 0:
                color = (0, 255, 0) if not eye.is_closed else (0, 0, 255)
                cv.circle(
                    frame,
                    (int(eye.center[0]), int(eye.center[1])),
                    int(eye.radius),
                    color,
                    2,
                )

        # Draw status text
        y_offset = 30
        cv.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, y_offset),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        y_offset += 30
        cv.putText(
            frame,
            f"Attention: {state.attention_score:.0%}",
            (10, y_offset),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if state.attention_score > 0.7 else (0, 0, 255),
            2,
        )

        # Alert banner
        if state.alert_level != AlertLevel.NORMAL:
            alert_colors = {
                AlertLevel.CAUTION: (0, 255, 255),
                AlertLevel.WARNING: (0, 165, 255),
                AlertLevel.CRITICAL: (0, 0, 255),
            }
            color = alert_colors[state.alert_level]

            cv.rectangle(frame, (0, h - 60), (w, h), color, -1)
            cv.putText(
                frame,
                f"ALERT: {state.alert_level.name}",
                (w // 2 - 100, h - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )


# Standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def clear_screen():
        """Clear terminal and move cursor to home position"""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def render_dashboard(dms, state, current_time):
        """Render the static dashboard"""
        # Move cursor to home position (top-left)
        sys.stdout.write("\033[H")

        # Build the entire output as a string
        output = []
        output.append("="*100)
        gpu_info = f"Physical GPU {selected_gpu_id}" if selected_gpu_id >= 0 else "CPU"
        output.append(f"[TIMESTAMP: {current_time:.2f}] | FPS: {dms.fps:.1f} | Model: {dms.model_name} ({dms.backend}) | Device: {gpu_info}")
        output.append("="*100)

        # Detection and Calibration Status
        output.append(f"\n┌─ SYSTEM STATUS {'─'*80}┐")
        face_status = "✓ DETECTED" if state.face_detected else "✗ NO FACE DETECTED"
        eyes_status = "✓ DETECTED" if state.eyes_detected else "✗ NO EYES DETECTED"
        cal_status = "✓ CALIBRATED" if dms.calibration["calibrated"] else "⚠ NOT CALIBRATED"
        output.append(f"│ Face: {face_status:<20} | Eyes: {eyes_status:<20} | Calibration: {cal_status:<20} │")
        output.append(f"└{'─'*98}┘")

        # Warning if not calibrated
        if not dms.calibration["calibrated"]:
            output.append("\n⚠ WARNING: System not calibrated! Press 'c' to calibrate for accurate detection.\n")

        # Eye Metrics
        output.append(f"\n┌─ EYE METRICS {'─'*82}┐")
        left_closed = 'YES' if state.left_eye.is_closed else 'NO'
        right_closed = 'YES' if state.right_eye.is_closed else 'NO'

        output.append(f"│ LEFT EYE:  Center=({state.left_eye.center[0]:.1f}, {state.left_eye.center[1]:.1f}) | "
                      f"Radius={state.left_eye.radius:.1f}px | "
                      f"Aspect Ratio={state.left_eye.aspect_ratio:.3f} | "
                      f"Closed={left_closed:<3}  │")
        output.append(f"│            Iris Pos=({state.left_eye.iris_position[0]:+.2f}, {state.left_eye.iris_position[1]:+.2f}) (normalized)"
                      + " "*40 + "│")
        output.append("│" + " "*98 + "│")
        output.append(f"│ RIGHT EYE: Center=({state.right_eye.center[0]:.1f}, {state.right_eye.center[1]:.1f}) | "
                      f"Radius={state.right_eye.radius:.1f}px | "
                      f"Aspect Ratio={state.right_eye.aspect_ratio:.3f} | "
                      f"Closed={right_closed:<3}  │")
        output.append(f"│            Iris Pos=({state.right_eye.iris_position[0]:+.2f}, {state.right_eye.iris_position[1]:+.2f}) (normalized)"
                      + " "*40 + "│")
        output.append("│" + " "*98 + "│")
        output.append(f"│ BLINK INFO: Eyes Closed Duration={state.eyes_closed_duration_ms:.0f}ms | "
                      f"Blink Rate={state.blink_rate:.1f} blinks/min" + " "*37 + "│")
        output.append(f"└{'─'*98}┘")

        # Head Pose & Gaze
        output.append(f"\n┌─ HEAD POSE & GAZE {'─'*76}┐")
        output.append(f"│ HEAD POSE: Yaw={state.head_pose[0]:+6.1f}° | "
                      f"Pitch={state.head_pose[1]:+6.1f}° | "
                      f"Roll={state.head_pose[2]:+6.1f}°" + " "*39 + "│")
        output.append(f"│ GAZE VEC:  X={state.gaze_vector[0]:+.3f} | "
                      f"Y={state.gaze_vector[1]:+.3f} | "
                      f"Z={state.gaze_vector[2]:+.3f} (forward=+Z)" + " "*35 + "│")
        output.append(f"└{'─'*98}┘")

        # Alert Scores
        output.append(f"\n┌─ ALERT SCORES {'─'*82}┐")

        # Handle None values when face not detected
        if state.attention_score is not None:
            attention_bar = '█' * int(state.attention_score * 20)
            attention_val = f"{state.attention_score*100:5.1f}%"
            attention_status = 'GOOD' if state.attention_score > 0.7 else 'LOW'
        else:
            attention_bar = ''
            attention_val = "  N/A"
            attention_status = 'N/A'

        if state.drowsiness_score is not None:
            drowsiness_bar = '█' * int(state.drowsiness_score * 20)
            drowsiness_val = f"{state.drowsiness_score*100:5.1f}%"
            drowsiness_status = 'ALERT' if state.drowsiness_score > 0.5 else 'OK'
        else:
            drowsiness_bar = ''
            drowsiness_val = "  N/A"
            drowsiness_status = 'N/A'

        if state.distraction_score is not None:
            distraction_bar = '█' * int(state.distraction_score * 20)
            distraction_val = f"{state.distraction_score*100:5.1f}%"
            distraction_status = 'ALERT' if state.distraction_score > 0.5 else 'OK'
        else:
            distraction_bar = ''
            distraction_val = "  N/A"
            distraction_status = 'N/A'

        output.append(f"│ ATTENTION:    {attention_val} {attention_bar:<20} "
                      f"[{attention_status:<5}]" + " "*30 + "│")
        output.append(f"│ DROWSINESS:   {drowsiness_val} {drowsiness_bar:<20} "
                      f"[{drowsiness_status:<5}]" + " "*30 + "│")
        output.append(f"│ DISTRACTION:  {distraction_val} {distraction_bar:<20} "
                      f"[{distraction_status:<5}]" + " "*30 + "│")
        alert_suffix = '[!!!]' if state.alert_level.value > 1 else '     '
        output.append(f"│ ALERT LEVEL:  {state.alert_level.name:<12} {alert_suffix}" + " "*62 + "│")
        output.append(f"└{'─'*98}┘")

        # Event Flags
        events = []
        if state.microsleep_detected:
            events.append("⚠ MICROSLEEP DETECTED")
        if state.looking_at_phone:
            events.append("⚠ LOOKING AT PHONE ZONE")
        if state.head_down_event:
            events.append("⚠ HEAD DOWN EVENT")

        output.append(f"\n┌─ ACTIVE EVENTS {'─'*81}┐")
        if events:
            for event in events:
                output.append(f"│ {event:<96} │")
        else:
            output.append(f"│ {'No active events':<96} │")
        output.append(f"└{'─'*98}┘")

        output.append("\n" + " "*30 + "Press 'q' to quit | Press 'c' for calibration")

        # Write all lines and clear to end of screen
        sys.stdout.write("\n".join(output))
        sys.stdout.write("\033[J")  # Clear from cursor to end of screen
        sys.stdout.flush()

    # Initialize DMS (GPU already configured via CUDA_VISIBLE_DEVICES)
    if selected_gpu_id >= 0:
        print(f"\nInitializing DMS on Physical GPU {selected_gpu_id}...")
        print(f"CUDA_VISIBLE_DEVICES={selected_gpu_id} (will appear as device 0 to frameworks)")
    else:
        print("\nInitializing DMS on CPU...")

    dms = DMS(camera_index=0)
    dms.start()
    last_print_time = 0
    output_interval = 0.5  # seconds

    # Initial screen setup
    clear_screen()
    print("\n" + "="*100)
    print("DMS MONITORING - Detailed Output Mode")
    print("="*100)
    if selected_gpu_id >= 0:
        print(f"GPU: Physical GPU {selected_gpu_id} (Logical Device 0)")
    else:
        print("GPU: CPU Mode")
    print("Starting monitoring...")
    print("="*100)
    time.sleep(2)  # Give user a moment to see startup message
    clear_screen()

    try:
        while True:
            # Get debug frame with overlay
            frame = dms.get_debug_frame()
            if frame is not None:
                cv.imshow("DMS Debug", frame)

            # Get latest state
            state = dms.get_latest_state()
            current_time = time.time()
            if state:
                # Update display every 0.5 seconds
                if current_time - last_print_time >= output_interval:
                    render_dashboard(dms, state, current_time)
                    last_print_time = current_time

            key = cv.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Calibrate
                clear_screen()
                dms.calibrate(duration_seconds=5)
                clear_screen()

    finally:
        dms.stop()
        cv.destroyAllWindows()
