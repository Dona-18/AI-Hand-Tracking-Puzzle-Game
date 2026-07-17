import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Game States
STATE_START = 0
STATE_COUNTDOWN = 1
STATE_PLAYING = 2
STATE_COMPLETED = 3


class Config:
    """Configurable constants for tuning sensitivity, layouts, and system configurations."""
    # Window settings
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    WINDOW_NAME = "AI Hand Tracking Puzzle Game"
    
    # Board size (Target dimensions for the puzzle image)
    BOARD_SIZE = 450
    GRID_SIZE = 3  # Dynamically creates a grid_size x grid_size puzzle (e.g. 3x3, 4x4)
    
    # Hand tracking & gesture parameters
    PINCH_THRESHOLD = 35        # Max pixel distance between thumb and index tip to pinch
    SNAP_DISTANCE = 40          # Distance in pixels to snap a piece to its target slot
    GESTURE_HOLD_DURATION = 1.2 # Time in seconds to hold open palm / fist for triggers
    
    # MediaPipe Hands confidence thresholds
    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE = 0.7
    
    # Camera hardware index
    CAMERA_INDEX = 0


@dataclass
class PuzzlePiece:
    """Dataclass holding all state information for a single puzzle piece."""
    piece_id: int
    image: np.ndarray
    correct_row: int
    correct_col: int
    correct_x: int
    correct_y: int
    current_x: int
    current_y: int
    w: int
    h: int
    is_placed: bool = False


class GestureDetector:
    """Handles MediaPipe Hand detection and maps hand landmarks to predefined game gestures."""
    def __init__(self):
        # Ensure hand landmarker model is downloaded
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise RuntimeError(f"Could not download hand_landmarker.task: {e}")
        
        # Configure detector options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=Config.TRACKING_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Gesture hold timers
        self.gesture_hold_start: Optional[float] = None
        self.current_held_gesture: Optional[str] = None
        
    def find_hand_landmarks(self, frame_rgb: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Runs the MediaPipe pipeline and extracts landmark pixel coordinates."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        if not detection_result.hand_landmarks:
            return None
        
        landmarks = []
        # Process only the first hand detected
        hand_lms = detection_result.hand_landmarks[0]
        for lm in hand_lms:
            px = int(lm.x * Config.WINDOW_WIDTH)
            py = int(lm.y * Config.WINDOW_HEIGHT)
            landmarks.append((px, py))
            
        return landmarks

    def get_gesture(self, lm: List[Tuple[int, int]]) -> str:
        """Determines the current active gesture from hand landmark positions."""
        def get_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            
        # 1. Pinch detection (highest priority gesture for tracking piece drag)
        pinch_dist = get_dist(lm[4], lm[8])
        if pinch_dist < Config.PINCH_THRESHOLD:
            return "PINCH"
            
        # Check extensions of fingers relative to joints (Y axis check in mirrored space)
        index_ext = lm[8][1] < lm[6][1]
        middle_ext = lm[12][1] < lm[10][1]
        ring_ext = lm[16][1] < lm[14][1]
        pinky_ext = lm[20][1] < lm[18][1]
        
        # Thumb: extended if distance from thumb tip (4) to middle MCP (9) is greater than IP (3) to middle MCP (9)
        thumb_ext = get_dist(lm[4], lm[9]) > get_dist(lm[3], lm[9])
        
        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext, thumb_ext])
        
        # 2. Fist: all fingers closed
        if extended_count == 0:
            return "FIST"
            
        # 3. Open Palm: at least 4 fingers extended
        if extended_count >= 4:
            return "OPEN PALM"
            
        # 4. Point: index is extended, middle/ring/pinky are folded
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "POINT"
            
        return "NONE"

    def update_hold_gesture(self, gesture: str) -> Tuple[Optional[str], float]:
        """
        Manages timer thresholds for held trigger gestures (Fist, Open Palm).
        Returns a tuple: (triggered_gesture_name, progress_ratio_from_0_to_1)
        """
        now = time.monotonic()
        
        if gesture in ["OPEN PALM", "FIST"]:
            if self.current_held_gesture == gesture:
                if self.gesture_hold_start is None:
                    self.gesture_hold_start = now
                elapsed = now - self.gesture_hold_start
                progress = min(1.0, elapsed / Config.GESTURE_HOLD_DURATION)
                if elapsed >= Config.GESTURE_HOLD_DURATION:
                    # Hold threshold triggered
                    self.current_held_gesture = None
                    self.gesture_hold_start = None
                    return gesture, 1.0
                return None, progress
            else:
                self.current_held_gesture = gesture
                self.gesture_hold_start = now
                return None, 0.0
        else:
            self.current_held_gesture = None
            self.gesture_hold_start = None
            return None, 0.0


class PuzzleGame:
    """Core game class holding video stream states, visual frames, rendering, and logic loops."""
    def __init__(self):
        self.state = STATE_START
        self.countdown_start = 0.0
        
        # Dynamically size pieces to match Config parameters cleanly
        self.grid_size = Config.GRID_SIZE
        self.cell_w = Config.BOARD_SIZE // self.grid_size
        self.cell_h = Config.BOARD_SIZE // self.grid_size
        self.board_width = self.cell_w * self.grid_size
        self.board_height = self.cell_h * self.grid_size
        
        # Centered frame preview box
        self.crop_x = (Config.WINDOW_WIDTH - self.board_width) // 2
        self.crop_y = (Config.WINDOW_HEIGHT - self.board_height) // 2
        
        # Target Board coordinates (right-aligned layout)
        self.board_x_min = 750
        self.board_y_min = self.crop_y
        self.board_x_max = self.board_x_min + self.board_width
        self.board_y_max = self.board_y_min + self.board_height
        
        self.captured_image: Optional[np.ndarray] = None
        self.pieces: List[PuzzlePiece] = []
        
        # Drag mechanics states
        self.dragged_piece: Optional[PuzzlePiece] = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        
        self.current_frame: Optional[np.ndarray] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def shuffle_pieces(self) -> None:
        """Scatters all unplaced puzzle pieces randomly on the left side of the screen."""
        for piece in self.pieces:
            # Random position within left side limits: x in [50, 700 - piece_w], y in [100, 680 - piece_h]
            piece.current_x = random.randint(50, 700 - self.cell_w - 50)
            piece.current_y = random.randint(100, 720 - self.cell_h - 50)
            piece.is_placed = False
        self.dragged_piece = None

    def reset_game(self) -> None:
        """Clears current puzzle image and transitions state machine back to webcam capture."""
        self.state = STATE_START
        self.captured_image = None
        self.pieces.clear()
        self.dragged_piece = None

    def capture_and_build_puzzle(self) -> None:
        """Crops a center square from the active webcam frame and slices it into grid pieces."""
        if self.current_frame is not None:
            # Slice crop region directly out of webcam buffer
            self.captured_image = self.current_frame[
                self.crop_y : self.crop_y + self.board_height,
                self.crop_x : self.crop_x + self.board_width
            ].copy()
            
            self.pieces.clear()
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    piece_img = self.captured_image[
                        row * self.cell_h : (row + 1) * self.cell_h,
                        col * self.cell_w : (col + 1) * self.cell_w
                    ].copy()
                    
                    correct_x = self.board_x_min + col * self.cell_w
                    correct_y = self.board_y_min + row * self.cell_h
                    
                    piece = PuzzlePiece(
                        piece_id=len(self.pieces),
                        image=piece_img,
                        correct_row=row,
                        correct_col=col,
                        correct_x=correct_x,
                        correct_y=correct_y,
                        current_x=correct_x, # Will be randomized immediately after
                        current_y=correct_y,
                        w=self.cell_w,
                        h=self.cell_h
                    )
                    self.pieces.append(piece)
            
            self.shuffle_pieces()
            self.state = STATE_PLAYING

    def update_game_logic(self, gesture: str, cursor_pos: Optional[Tuple[int, int]]) -> None:
        """Executes core game loops, piece tracking, snapping math, and transitions."""
        if self.state == STATE_COUNTDOWN:
            elapsed = time.monotonic() - self.countdown_start
            if elapsed >= 3.0:
                self.capture_and_build_puzzle()
                
        elif self.state == STATE_PLAYING:
            if gesture == "PINCH" and cursor_pos:
                cx, cy = cursor_pos
                if self.dragged_piece is None:
                    # Look for pieces under the cursor. Loop reversed to pick up top-most piece first
                    for piece in reversed(self.pieces):
                        if not piece.is_placed:
                            if piece.current_x <= cx <= piece.current_x + piece.w and \
                               piece.current_y <= cy <= piece.current_y + piece.h:
                                self.dragged_piece = piece
                                self.drag_offset_x = cx - piece.current_x
                                self.drag_offset_y = cy - piece.current_y
                                
                                # Move grabbed piece to the end of list so it renders on top
                                self.pieces.remove(piece)
                                self.pieces.append(piece)
                                break
                else:
                    # Move currently dragged piece with cursor tracking offset
                    self.dragged_piece.current_x = cx - self.drag_offset_x
                    self.dragged_piece.current_y = cy - self.drag_offset_y
                    
                    # Safe boundaries clamp: Keep piece within frame dimensions safely
                    self.dragged_piece.current_x = max(0, min(self.dragged_piece.current_x, Config.WINDOW_WIDTH - self.dragged_piece.w))
                    self.dragged_piece.current_y = max(0, min(self.dragged_piece.current_y, Config.WINDOW_HEIGHT - self.dragged_piece.h))
            else:
                # Gesture is not a PINCH, handle releases
                if self.dragged_piece is not None:
                    p = self.dragged_piece
                    # Calculate Euclidean distance to destination coordinates
                    dist = math.hypot(p.current_x - p.correct_x, p.current_y - p.correct_y)
                    if dist < Config.SNAP_DISTANCE:
                        p.current_x = p.correct_x
                        p.current_y = p.correct_y
                        p.is_placed = True
                    self.dragged_piece = None
                    
                # Verify complete board completion
                if len(self.pieces) > 0 and all(piece.is_placed for piece in self.pieces):
                    self.state = STATE_COMPLETED

    def draw_panel(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float) -> None:
        """Helper to render high-contrast, semi-transparent window backing blocks."""
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    def draw_text(self, img: np.ndarray, text: str, pos: Tuple[int, int], scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
        """Draws readable text characters utilizing background drop-shadows."""
        cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_dashed_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1, gap: int = 15) -> None:
        """Custom renderer to draw high-fidelity dashed rectangular segments."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Draw horizontal segmented lines
        for x in range(x1, x2, gap * 2):
            cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness, cv2.LINE_AA)
        # Draw vertical segmented lines
        for y in range(y1, y2, gap * 2):
            cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness, cv2.LINE_AA)

    def draw_piece_safely(self, frame: np.ndarray, piece: PuzzlePiece, border_color: Tuple[int, int, int], border_thickness: int) -> None:
        """Safely overlays puzzle piece textures using boundary clipping to prevent NumPy slice errors."""
        x1 = max(0, min(piece.current_x, Config.WINDOW_WIDTH))
        y1 = max(0, min(piece.current_y, Config.WINDOW_HEIGHT))
        x2 = max(0, min(piece.current_x + piece.w, Config.WINDOW_WIDTH))
        y2 = max(0, min(piece.current_y + piece.h, Config.WINDOW_HEIGHT))
        
        # Calculate local sub-image offsets in case piece is partially offscreen
        img_x1 = x1 - piece.current_x
        img_y1 = y1 - piece.current_y
        img_x2 = img_x1 + (x2 - x1)
        img_y2 = img_y1 + (y2 - y1)
        
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            frame[y1:y2, x1:x2] = piece.image[img_y1:img_y2, img_x1:img_x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

    def draw_skeleton(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> None:
        """Draws a premium holographic skeleton overlay mapping the user's hand landmarks."""
        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (9, 10), (10, 11), (11, 12),           # Middle
            (13, 14), (14, 15), (15, 16),          # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)              # Knuckles
        ]
        
        overlay = frame.copy()
        for start_idx, end_idx in CONNECTIONS:
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            cv2.line(overlay, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA) # Cyan connections
            
        for i, pt in enumerate(landmarks):
            if i == 8:
                cv2.circle(overlay, pt, 7, (0, 255, 255), -1, cv2.LINE_AA) # Yellow index tip (Cursor)
            elif i == 4:
                cv2.circle(overlay, pt, 7, (0, 255, 0), -1, cv2.LINE_AA)   # Green thumb tip
            else:
                cv2.circle(overlay, pt, 4, (180, 105, 255), -1, cv2.LINE_AA) # Soft purple joints
                
        # Draw translucent alpha blend
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    def render(self, frame: np.ndarray, landmarks: Optional[List[Tuple[int, int]]], gesture: str, hold_progress: float, fps: float) -> None:
        """Renders current visual templates, headers, puzzle pieces, overlays, and custom UI components."""
        # 1. Dark background paneling to highlight workspace
        if self.state in [STATE_PLAYING, STATE_COMPLETED]:
            self.draw_panel(frame, 0, 0, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT, (25, 25, 25), 0.4)
            
        # 2. Hand landmarks skeleton
        if landmarks:
            self.draw_skeleton(frame, landmarks)
            
        # 3. State-specific content rendering
        if self.state == STATE_START:
            # Alignment rectangle
            self.draw_dashed_rect(frame, (self.crop_x, self.crop_y), (self.crop_x + self.board_width, self.crop_y + self.board_height), (255, 255, 255), 2)
            self.draw_text(frame, "CROP PREVIEW AREA", (self.crop_x + 105, self.crop_y - 15), 0.7, (255, 255, 255), 2)
            self.draw_text(frame, "Align your face / object here", (self.crop_x + 75, self.crop_y + self.board_height + 25), 0.6, (200, 200, 200), 1)
            
            # Sidebar instructions card
            self.draw_panel(frame, 40, self.crop_y, 375, self.crop_y + self.board_height, (20, 20, 20), 0.75)
            cv2.rectangle(frame, (40, self.crop_y), (375, self.crop_y + self.board_height), (80, 80, 80), 1)
            
            self.draw_text(frame, "HOW TO PLAY", (110, self.crop_y + 40), 0.8, (0, 255, 255), 2)
            
            y_offset = self.crop_y + 90
            instructions = [
                "1. Align camera subject",
                "   inside the center box.",
                "",
                "2. Hold an OPEN PALM",
                "   for 1.2s to capture",
                "   (or press 'C' key).",
                "",
                "3. Pinch (Thumb + Index)",
                "   to drag puzzle pieces.",
                "",
                "4. Drop a piece near its",
                "   slot to snap it.",
                "",
                "5. Hold FIST for 1.2s",
                "   to reshuffle anytime."
            ]
            for inst in instructions:
                self.draw_text(frame, inst, (60, y_offset), 0.5, (220, 220, 220), 1)
                y_offset += 22
                
        elif self.state == STATE_COUNTDOWN:
            # Preview box
            self.draw_dashed_rect(frame, (self.crop_x, self.crop_y), (self.crop_x + self.board_width, self.crop_y + self.board_height), (255, 255, 255), 2)
            
            elapsed = time.monotonic() - self.countdown_start
            time_left = max(0.0, 3.0 - elapsed)
            seconds = int(math.ceil(time_left))
            
            center_x, center_y = Config.WINDOW_WIDTH // 2, Config.WINDOW_HEIGHT // 2
            radius = 80
            # Center target background backing
            self.draw_panel(frame, center_x - radius, center_y - radius, center_x + radius, center_y + radius, (10, 10, 10), 0.6)
            cv2.circle(frame, (center_x, center_y), radius, (0, 200, 255), 3, cv2.LINE_AA)
            
            # Big text countdown
            self.draw_text(frame, str(seconds), (center_x - 22, center_y + 30), 3.5, (0, 255, 255), 7)
            
        elif self.state in [STATE_PLAYING, STATE_COMPLETED]:
            # Draw alpha phantom solutions background on target board
            if self.captured_image is not None:
                board_area = frame[self.board_y_min:self.board_y_max, self.board_x_min:self.board_x_max]
                cv2.addWeighted(self.captured_image, 0.18, board_area, 0.82, 0, dst=board_area)
                
            # Draw slot grid dividers
            for i in range(1, self.grid_size):
                cx = self.board_x_min + i * self.cell_w
                cv2.line(frame, (cx, self.board_y_min), (cx, self.board_y_max), (100, 100, 100), 1)
                cy = self.board_y_min + i * self.cell_h
                cv2.line(frame, (self.board_x_min, cy), (self.board_x_max, cy), (100, 100, 100), 1)
                
            # Board borders
            cv2.rectangle(frame, (self.board_x_min, self.board_y_min), (self.board_x_max, self.board_y_max), (180, 180, 180), 2)
            self.draw_text(frame, "PUZZLE BOARD", (self.board_x_min + 130, self.board_y_min - 15), 0.7, (200, 200, 200), 2)
            
            # Render pieces cleanly in logical layering order
            # Placed pieces: Solid green border
            for piece in self.pieces:
                if piece.is_placed:
                    self.draw_piece_safely(frame, piece, (100, 230, 100), 2)
            # Unplaced pieces: Thin white border
            for piece in self.pieces:
                if not piece.is_placed and piece != self.dragged_piece:
                    self.draw_piece_safely(frame, piece, (255, 255, 255), 1)
            # Active piece: Thick glowing cyan border
            if self.dragged_piece:
                self.draw_piece_safely(frame, self.dragged_piece, (0, 255, 255), 3)
                
            # Win screen victory popup card
            if self.state == STATE_COMPLETED:
                self.draw_panel(frame, 390, 200, 890, 520, (15, 35, 15), 0.8)
                cv2.rectangle(frame, (390, 200), (890, 520), (0, 230, 0), 2)
                
                self.draw_text(frame, "PUZZLE COMPLETED!", (435, 260), 0.9, (0, 255, 0), 3)
                self.draw_text(frame, "Outstanding! You solved the puzzle.", (445, 315), 0.6, (240, 240, 240), 1)
                self.draw_text(frame, "Hold a FIST for 1.2s to reset,", (475, 380), 0.55, (200, 200, 200), 1)
                self.draw_text(frame, "or press 'R' key to play again.", (465, 410), 0.55, (200, 200, 200), 1)
                self.draw_text(frame, "Press 'Q' or 'Esc' to exit.", (495, 470), 0.5, (0, 255, 255), 1)
                
        # 4. HUD Top Header Panel
        self.draw_panel(frame, 0, 0, Config.WINDOW_WIDTH, 80, (15, 15, 15), 0.8)
        cv2.line(frame, (0, 80), (Config.WINDOW_WIDTH, 80), (50, 50, 50), 1)
        
        self.draw_text(frame, "AI GESTURE PUZZLE", (30, 48), 0.8, (255, 255, 255), 2)
        
        # UI color-coded gestures
        gesture_color = (255, 255, 255)
        if gesture == "PINCH":
            gesture_color = (100, 255, 100) # Neon green
        elif gesture == "OPEN PALM":
            gesture_color = (0, 200, 255)   # Amber/Cyan
        elif gesture == "FIST":
            gesture_color = (100, 100, 255) # Soft Red-Purple
        elif gesture == "POINT":
            gesture_color = (255, 150, 50)  # Bright Orange
            
        self.draw_text(frame, f"GESTURE: {gesture}", (550, 48), 0.7, gesture_color, 2)
        
        placed_count = sum(1 for p in self.pieces if p.is_placed)
        total_pieces = len(self.pieces)
        
        if self.state in [STATE_PLAYING, STATE_COMPLETED]:
            status_color = (0, 255, 0) if placed_count == total_pieces else (255, 255, 255)
            self.draw_text(frame, f"PIECES: {placed_count} / {total_pieces}", (980, 48), 0.7, status_color, 2)
        else:
            self.draw_text(frame, "STATE: ALIGN & CAPTURE", (940, 48), 0.65, (255, 255, 255), 2)
            
        # Draw frame FPS overlay
        self.draw_text(frame, f"FPS: {int(fps)}", (1200, 25), 0.4, (150, 150, 150), 1)
        
        # 5. Bottom Instructions Panel
        self.draw_panel(frame, 0, Config.WINDOW_HEIGHT - 65, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT, (15, 15, 15), 0.8)
        cv2.line(frame, (0, Config.WINDOW_HEIGHT - 65), (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT - 65), (50, 50, 50), 1)
        
        self.draw_text(frame, "GESTURES: Open Palm (Hold 1.2s) = Capture  |  Pinch = Drag  |  Fist (Hold 1.2s) = Reshuffle/Reset", (30, Config.WINDOW_HEIGHT - 38), 0.45, (180, 180, 180), 1)
        self.draw_text(frame, "KEYS: C = Capture  |  S = Shuffle  |  R = Reset  |  Q = Quit", (880, Config.WINDOW_HEIGHT - 38), 0.45, (0, 255, 255), 1)
        
        # 6. Cursor rings & gesture timer progress bars
        if landmarks:
            cursor_x, cursor_y = landmarks[8]
            
            if gesture == "PINCH":
                tx, ty = landmarks[4]
                cv2.line(frame, (cursor_x, cursor_y), (tx, ty), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (cursor_x, cursor_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (cursor_x, cursor_y), 12, (0, 255, 0), 2, cv2.LINE_AA)
            elif gesture == "POINT":
                cv2.circle(frame, (cursor_x, cursor_y), 8, (255, 150, 50), 2, cv2.LINE_AA)
                cv2.circle(frame, (cursor_x, cursor_y), 3, (255, 150, 50), -1, cv2.LINE_AA)
                
            # Circle timer arc for hold states
            if hold_progress > 0.0:
                center = (cursor_x, cursor_y)
                radius = 22
                angle = int(hold_progress * 360)
                # Outer gray trace
                cv2.ellipse(frame, center, (radius, radius), -90, 0, 360, (80, 80, 80), 2, cv2.LINE_AA)
                # Active progress ring
                cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 165, 255), 3, cv2.LINE_AA)

    def run(self) -> None:
        """Main game control loop, camera handler, and clean system destruction manager."""
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            self.show_camera_error_window()
            return
            
        cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(Config.WINDOW_NAME, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        
        detector = GestureDetector()
        prev_time = time.monotonic()
        
        while True:
            ret, raw_frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Horizontal mirror flip
            frame = cv2.flip(raw_frame, 1)
            frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            # FPS tracking calculations
            curr_time = time.monotonic()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time
            
            # Convert BGR frames to RGB for mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.find_hand_landmarks(rgb_frame)
            
            gesture = "NONE"
            cursor_pos = None
            
            if landmarks:
                gesture = detector.get_gesture(landmarks)
                cursor_pos = landmarks[8] # Index fingertip
                
            # Parse timing events for held gestures
            triggered_gesture, hold_progress = detector.update_hold_gesture(gesture)
            
            # Execute events based on gesture thresholds
            if triggered_gesture == "OPEN PALM":
                if self.state == STATE_START:
                    self.state = STATE_COUNTDOWN
                    self.countdown_start = time.monotonic()
            elif triggered_gesture == "FIST":
                if self.state == STATE_PLAYING:
                    self.shuffle_pieces()
                elif self.state == STATE_COMPLETED:
                    self.reset_game()
            
            # Store the current mirror frame for cropping puzzle segments
            self.current_frame = frame.copy()
            
            # Handle game state updates
            self.update_game_logic(gesture, cursor_pos)
            
            # Draw frame overlays & assets
            self.render(frame, landmarks, gesture, hold_progress, fps)
            
            cv2.imshow(Config.WINDOW_NAME, frame)
            
            # Handle Keyboard overrides
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27: # Esc key is 27
                break
            elif key == ord('c') or key == ord('C'):
                if self.state == STATE_START:
                    self.state = STATE_COUNTDOWN
                    self.countdown_start = time.monotonic()
            elif key == ord('s') or key == ord('S'):
                if self.state == STATE_PLAYING:
                    self.shuffle_pieces()
            elif key == ord('r') or key == ord('R'):
                self.reset_game()
                
        # Clean resources
        self.cap.release()
        cv2.destroyAllWindows()

    def show_camera_error_window(self) -> None:
        """Displays error window with macOS system diagnostics when webcam can't be initialized."""
        error_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        error_img[:] = (30, 30, 30) # Dark Charcoal
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(error_img, "CAMERA ACCESS ERROR", (340, 230), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(error_img, "Could not open webcam. Please verify:", (300, 310), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(error_img, "1. The webcam is connected properly.", (300, 360), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(error_img, "2. Camera permissions are granted (macOS System Settings).", (300, 410), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(error_img, "3. Config.CAMERA_INDEX in app.py is set correctly.", (300, 460), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(error_img, "Press 'Q' or 'Esc' to exit.", (450, 560), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("AI Hand Tracking Puzzle Game - Camera Error", error_img)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = PuzzleGame()
    game.run()
