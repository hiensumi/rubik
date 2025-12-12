"""
Rubik's Cube Scanner - DearPyGui Version
=========================================
Complete scanner application using DearPyGui for all GUI elements.
"""

import cv2
import torch
import numpy as np
import kociemba
import dearpygui.dearpygui as dpg
import math
import copy
import time
from ultralytics import YOLO

# Import face reading module
from rubik_face_read import (
    CubeState, extract_face_lab, FACE_ORDER, apply_move
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "IMG_SIZE": 224,
}

# Color mapping to RGB (0-255) for DearPyGui
COLOR_RGB = {
    'white':  (255, 255, 255),
    'yellow': (255, 255, 0),
    'red':    (255, 0, 0),
    'orange': (255, 140, 0),
    'green':  (0, 200, 0),
    'blue':   (0, 100, 255),
    'unknown': (128, 128, 128),
}

# Move to face and axis mapping for animation
MOVE_INFO = {
    # Format: move -> (face_name, rotation_axis, direction)
    # direction: 1 = clockwise when looking at face, -1 = counter-clockwise
    'U': ('white', (0, 1, 0), -1),    # White face, Y axis
    "U'": ('white', (0, 1, 0), 1),
    'U2': ('white', (0, 1, 0), -1),   # Same direction, just 180 degrees
    'D': ('yellow', (0, 1, 0), 1),    # Yellow face, Y axis
    "D'": ('yellow', (0, 1, 0), -1),
    'D2': ('yellow', (0, 1, 0), -1),
    'F': ('green', (0, 0, 1), -1),    # Green face, Z axis
    "F'": ('green', (0, 0, 1), 1),
    'F2': ('green', (0, 0, 1), -1),
    'B': ('blue', (0, 0, 1), 1),      # Blue face, Z axis (opposite direction)
    "B'": ('blue', (0, 0, 1), -1),
    'B2': ('blue', (0, 0, 1), 1),
    'R': ('red', (1, 0, 0), -1),      # Red face, X axis
    "R'": ('red', (1, 0, 0), 1),
    'R2': ('red', (1, 0, 0), -1),
    'L': ('orange', (1, 0, 0), 1),    # Orange face, X axis (opposite direction)
    "L'": ('orange', (1, 0, 0), -1),
    'L2': ('orange', (1, 0, 0), 1),
}

# ============================================================================
# 3D CUBE RENDERING
# ============================================================================

def rotate_point_3d(x, y, z, rx, ry):
    """Rotate a 3D point around X and Y axes"""
    # First rotate around Y axis (yaw)
    cos_y = math.cos(math.radians(ry))
    sin_y = math.sin(math.radians(ry))
    x1 = x * cos_y + z * sin_y
    z1 = -x * sin_y + z * cos_y
    
    # Then rotate around X axis (pitch)
    cos_x = math.cos(math.radians(rx))
    sin_x = math.sin(math.radians(rx))
    y1 = y * cos_x - z1 * sin_x
    z2 = y * sin_x + z1 * cos_x
    
    return x1, y1, z2


def project_3d_to_2d(x, y, z, center_x, center_y, scale):
    """Project 3D point to 2D with perspective"""
    fov = 4.0
    z_offset = 6.0
    factor = fov / (z + z_offset)
    screen_x = center_x - x * scale * factor  # Flip X to correct horizontal mirror
    screen_y = center_y - y * scale * factor  # Y is flipped for screen coords
    return screen_x, screen_y, z


def get_cube_quads(faces, rx, ry, center_x, center_y, scale):
    """Get all face quads with colors for rendering - simplified approach"""
    quads = []
    
    # Define faces with their 3D positions
    # Each face: (name, center position, right direction (col+), up direction (row-))
    # These are set so when you look directly at each face, it appears correctly oriented
    face_defs = [
        # White (top) - looking down at it, green is at bottom, red is at right
        ('white',  (0, 1, 0),  (1, 0, 0),  (0, 0, -1)),
        # Yellow (bottom) - looking up at it, green is at top, red is at right  
        ('yellow', (0, -1, 0), (1, 0, 0),  (0, 0, 1)),
        # Green (front) - looking at it, red is at right, white is at top
        ('green',  (0, 0, 1),  (1, 0, 0),  (0, 1, 0)),
        # Blue (back) - looking at it from behind, orange is at right, white is at top
        ('blue',   (0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        # Red (right) - looking at it, blue is at right, white is at top
        ('red',    (1, 0, 0),  (0, 0, -1), (0, 1, 0)),
        # Orange (left) - looking at it, green is at right, white is at top
        ('orange', (-1, 0, 0), (0, 0, 1),  (0, 1, 0)),
    ]
    
    for face_name, center, right, up in face_defs:
        face = faces.get(face_name)
        if face is None:
            continue
        
        cx, cy, cz = center
        rx_dir, ry_dir, rz_dir = right
        ux, uy, uz = up
        
        for row in range(3):
            for col in range(3):
                sticker_color = face[row][col]
                rgb = COLOR_RGB.get(sticker_color, COLOR_RGB['unknown'])
                
                # Position on face: col goes along 'right', row goes opposite to 'up'
                u = (col - 1) * 0.65  # -1, 0, 1 for columns
                v = (1 - row) * 0.65  # 1, 0, -1 for rows (top to bottom)
                
                # Sticker corners
                corners_3d = []
                hs = 0.28  # Half sticker size
                
                for du, dv in [(-hs, hs), (hs, hs), (hs, -hs), (-hs, -hs)]:
                    # 3D position
                    px = cx + (u + du) * rx_dir + (v + dv) * ux
                    py = cy + (u + du) * ry_dir + (v + dv) * uy
                    pz = cz + (u + du) * rz_dir + (v + dv) * uz
                    
                    # Rotate and project
                    rpx, rpy, rpz = rotate_point_3d(px, py, pz, rx, ry)
                    sx, sy, sz = project_3d_to_2d(rpx, rpy, rpz, center_x, center_y, scale)
                    corners_3d.append((sx, sy, sz))
                
                avg_z = sum(c[2] for c in corners_3d) / 4
                
                # Backface culling using winding order
                # Cross product determines if vertices are clockwise or counter-clockwise
                e1x = corners_3d[1][0] - corners_3d[0][0]
                e1y = corners_3d[1][1] - corners_3d[0][1]
                e2x = corners_3d[2][0] - corners_3d[0][0]
                e2y = corners_3d[2][1] - corners_3d[0][1]
                cross = e1x * e2y - e1y * e2x
                
                # Add all quads with their winding info
                corners_2d = [(c[0], c[1]) for c in corners_3d]
                quads.append((avg_z, corners_2d, rgb, cross))
    
    quads.sort(key=lambda q: -q[0])
    return quads


def draw_cube_on_canvas(canvas_tag, faces, rx, ry, width, height):
    """Draw 3D cube on a DearPyGui canvas"""
    dpg.delete_item(canvas_tag, children_only=True)
    
    center_x = width / 2
    center_y = height / 2
    scale = min(width, height) / 4.5
    
    # Draw background
    dpg.draw_rectangle((0, 0), (width, height), fill=(25, 25, 30, 255), parent=canvas_tag)
    
    quads = get_cube_quads(faces, rx, ry, center_x, center_y, scale)
    
    for item in quads:
        avg_z, corners, rgb, cross = item
        # Draw front-facing quads (cross > 0 means CCW winding = front facing)
        if cross > 0:
            dpg.draw_quad(corners[0], corners[1], corners[2], corners[3],
                         color=(20, 20, 25, 255),
                         fill=(*rgb, 255),
                         thickness=2,
                         parent=canvas_tag)


def rotate_point_around_axis(px, py, pz, axis, angle_deg):
    """Rotate a point around an axis through origin using Rodrigues' formula"""
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    ax, ay, az = axis
    
    # Rodrigues' rotation formula
    # v_rot = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
    # where k is the unit axis vector
    
    # k x v (cross product)
    cross_x = ay * pz - az * py
    cross_y = az * px - ax * pz
    cross_z = ax * py - ay * px
    
    # k . v (dot product)
    dot = ax * px + ay * py + az * pz
    
    # Result
    new_x = px * cos_a + cross_x * sin_a + ax * dot * (1 - cos_a)
    new_y = py * cos_a + cross_y * sin_a + ay * dot * (1 - cos_a)
    new_z = pz * cos_a + cross_z * sin_a + az * dot * (1 - cos_a)
    
    return new_x, new_y, new_z


def get_stickers_for_layer(face_name):
    """Get which stickers are part of a rotating layer.
    Returns list of (face, row, col) for all stickers that rotate together.
    This includes the face itself plus the adjacent edge row/col on neighboring faces.
    """
    # The face being rotated + adjacent edges on neighbor faces
    layer_stickers = {
        # U (white) - top layer: white face + top row of F/R/B/L
        'white': [
            ('white', 0, 0), ('white', 0, 1), ('white', 0, 2),
            ('white', 1, 0), ('white', 1, 1), ('white', 1, 2),
            ('white', 2, 0), ('white', 2, 1), ('white', 2, 2),
            ('green', 0, 0), ('green', 0, 1), ('green', 0, 2),
            ('red', 0, 0), ('red', 0, 1), ('red', 0, 2),
            ('blue', 0, 0), ('blue', 0, 1), ('blue', 0, 2),
            ('orange', 0, 0), ('orange', 0, 1), ('orange', 0, 2),
        ],
        # D (yellow) - bottom layer: yellow face + bottom row of F/R/B/L
        'yellow': [
            ('yellow', 0, 0), ('yellow', 0, 1), ('yellow', 0, 2),
            ('yellow', 1, 0), ('yellow', 1, 1), ('yellow', 1, 2),
            ('yellow', 2, 0), ('yellow', 2, 1), ('yellow', 2, 2),
            ('green', 2, 0), ('green', 2, 1), ('green', 2, 2),
            ('red', 2, 0), ('red', 2, 1), ('red', 2, 2),
            ('blue', 2, 0), ('blue', 2, 1), ('blue', 2, 2),
            ('orange', 2, 0), ('orange', 2, 1), ('orange', 2, 2),
        ],
        # F (green) - front layer: green face + adjacent edges
        'green': [
            ('green', 0, 0), ('green', 0, 1), ('green', 0, 2),
            ('green', 1, 0), ('green', 1, 1), ('green', 1, 2),
            ('green', 2, 0), ('green', 2, 1), ('green', 2, 2),
            ('white', 2, 0), ('white', 2, 1), ('white', 2, 2),
            ('red', 0, 0), ('red', 1, 0), ('red', 2, 0),
            ('yellow', 0, 0), ('yellow', 0, 1), ('yellow', 0, 2),
            ('orange', 0, 2), ('orange', 1, 2), ('orange', 2, 2),
        ],
        # B (blue) - back layer: blue face + adjacent edges
        'blue': [
            ('blue', 0, 0), ('blue', 0, 1), ('blue', 0, 2),
            ('blue', 1, 0), ('blue', 1, 1), ('blue', 1, 2),
            ('blue', 2, 0), ('blue', 2, 1), ('blue', 2, 2),
            ('white', 0, 0), ('white', 0, 1), ('white', 0, 2),
            ('orange', 0, 0), ('orange', 1, 0), ('orange', 2, 0),
            ('yellow', 2, 0), ('yellow', 2, 1), ('yellow', 2, 2),
            ('red', 0, 2), ('red', 1, 2), ('red', 2, 2),
        ],
        # R (red) - right layer: red face + right col of U/F/D/B
        'red': [
            ('red', 0, 0), ('red', 0, 1), ('red', 0, 2),
            ('red', 1, 0), ('red', 1, 1), ('red', 1, 2),
            ('red', 2, 0), ('red', 2, 1), ('red', 2, 2),
            ('white', 0, 2), ('white', 1, 2), ('white', 2, 2),
            ('green', 0, 2), ('green', 1, 2), ('green', 2, 2),
            ('yellow', 0, 2), ('yellow', 1, 2), ('yellow', 2, 2),
            ('blue', 0, 0), ('blue', 1, 0), ('blue', 2, 0),
        ],
        # L (orange) - left layer: orange face + left col of U/F/D/B
        'orange': [
            ('orange', 0, 0), ('orange', 0, 1), ('orange', 0, 2),
            ('orange', 1, 0), ('orange', 1, 1), ('orange', 1, 2),
            ('orange', 2, 0), ('orange', 2, 1), ('orange', 2, 2),
            ('white', 0, 0), ('white', 1, 0), ('white', 2, 0),
            ('green', 0, 0), ('green', 1, 0), ('green', 2, 0),
            ('yellow', 0, 0), ('yellow', 1, 0), ('yellow', 2, 0),
            ('blue', 0, 2), ('blue', 1, 2), ('blue', 2, 2),
        ],
    }
    return layer_stickers.get(face_name, [])


def draw_animated_cube(canvas_tag, faces, rx, ry, width, height, rotating_face=None, rotation_angle=0):
    """Draw 3D cube with animated face rotation"""
    dpg.delete_item(canvas_tag, children_only=True)
    
    center_x = width / 2
    center_y = height / 2
    scale = min(width, height) / 4.5
    
    # Draw background
    dpg.draw_rectangle((0, 0), (width, height), fill=(25, 25, 30, 255), parent=canvas_tag)
    
    quads = []
    
    # Get stickers that should rotate
    rotating_stickers = set()
    rotation_axis = (0, 0, 0)
    if rotating_face and rotating_face in MOVE_INFO:
        face_name, axis, direction = MOVE_INFO[rotating_face]
        rotation_axis = axis
        for sticker in get_stickers_for_layer(face_name):
            rotating_stickers.add(sticker)
    
    # Face definitions
    face_defs = [
        ('white',  (0, 1, 0),  (1, 0, 0),  (0, 0, -1)),
        ('yellow', (0, -1, 0), (1, 0, 0),  (0, 0, 1)),
        ('green',  (0, 0, 1),  (1, 0, 0),  (0, 1, 0)),
        ('blue',   (0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ('red',    (1, 0, 0),  (0, 0, -1), (0, 1, 0)),
        ('orange', (-1, 0, 0), (0, 0, 1),  (0, 1, 0)),
    ]
    
    for face_name, center, right, up in face_defs:
        face = faces.get(face_name)
        if face is None:
            continue
        
        cx, cy, cz = center
        rx_dir, ry_dir, rz_dir = right
        ux, uy, uz = up
        
        for row in range(3):
            for col in range(3):
                sticker_color = face[row][col]
                rgb = COLOR_RGB.get(sticker_color, COLOR_RGB['unknown'])
                
                u = (col - 1) * 0.65
                v = (1 - row) * 0.65
                
                corners_3d = []
                hs = 0.28
                
                # Check if this sticker should rotate
                is_rotating = (face_name, row, col) in rotating_stickers
                
                for du, dv in [(-hs, hs), (hs, hs), (hs, -hs), (-hs, -hs)]:
                    px = cx + (u + du) * rx_dir + (v + dv) * ux
                    py = cy + (u + du) * ry_dir + (v + dv) * uy
                    pz = cz + (u + du) * rz_dir + (v + dv) * uz
                    
                    # Apply layer rotation if this sticker is part of rotating layer
                    if is_rotating and rotation_angle != 0:
                        px, py, pz = rotate_point_around_axis(px, py, pz, rotation_axis, rotation_angle)
                    
                    rpx, rpy, rpz = rotate_point_3d(px, py, pz, rx, ry)
                    sx, sy, sz = project_3d_to_2d(rpx, rpy, rpz, center_x, center_y, scale)
                    corners_3d.append((sx, sy, sz))
                
                avg_z = sum(c[2] for c in corners_3d) / 4
                
                e1x = corners_3d[1][0] - corners_3d[0][0]
                e1y = corners_3d[1][1] - corners_3d[0][1]
                e2x = corners_3d[2][0] - corners_3d[0][0]
                e2y = corners_3d[2][1] - corners_3d[0][1]
                cross = e1x * e2y - e1y * e2x
                
                corners_2d = [(c[0], c[1]) for c in corners_3d]
                quads.append((avg_z, corners_2d, rgb, cross))
    
    quads.sort(key=lambda q: -q[0])
    
    for item in quads:
        avg_z, corners, rgb, cross = item
        if cross > 0:
            dpg.draw_quad(corners[0], corners[1], corners[2], corners[3],
                         color=(20, 20, 25, 255),
                         fill=(*rgb, 255),
                         thickness=2,
                         parent=canvas_tag)


def draw_flat_cube_preview(canvas_tag, cube_state, width, height):
    """Draw flattened cube net on a DearPyGui canvas
    
    Layout:
            [W]
        [O] [G] [R] [B]
            [Y]
    """
    dpg.delete_item(canvas_tag, children_only=True)
    
    # Draw background
    dpg.draw_rectangle((0, 0), (width, height), fill=(25, 25, 30, 255), parent=canvas_tag)
    
    # Calculate cell size to fit 4 faces wide, 3 faces tall
    cell_size = min(width // 12, height // 9)
    face_size = cell_size * 3
    
    # Offsets to center the layout
    total_w = face_size * 4 + 6
    total_h = face_size * 3 + 4
    offset_x = (width - total_w) // 2
    offset_y = (height - total_h) // 2
    
    # Face positions: (col, row) in the unfolded layout
    face_positions = {
        'white':  (1, 0),  # Top
        'orange': (0, 1),  # Left
        'green':  (1, 1),  # Front
        'red':    (2, 1),  # Right
        'blue':   (3, 1),  # Back
        'yellow': (1, 2),  # Bottom
    }
    
    for face_name, (fx, fy) in face_positions.items():
        face = cube_state.faces.get(face_name)
        base_x = offset_x + fx * (face_size + 2)
        base_y = offset_y + fy * (face_size + 2)
        
        if face is None:
            # Draw empty placeholder
            dpg.draw_rectangle((base_x, base_y), (base_x + face_size, base_y + face_size),
                              color=(80, 80, 90, 255), thickness=1, parent=canvas_tag)
            continue
        
        for row in range(3):
            for col in range(3):
                x = base_x + col * cell_size
                y = base_y + row * cell_size
                
                sticker_color = face[row][col]
                rgb = COLOR_RGB.get(sticker_color, COLOR_RGB['unknown'])
                
                # Draw filled sticker
                dpg.draw_rectangle((x + 1, y + 1), (x + cell_size - 1, y + cell_size - 1),
                                  fill=(*rgb, 255), color=(0, 0, 0, 255), 
                                  thickness=1, parent=canvas_tag)


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class RubikScannerApp:
    def __init__(self):
        self.img_size = 224
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO("./YOLO.pt")
        print("YOLO model loaded.")

        # State
        self.cube_state = CubeState()
        self.solution = None
        self.validation_errors = []  # Store validation errors for GUI display
        self.cap = None
        self.running = False
        self.last_warped = None
        self.is_present = False
        self.presence_prob = 0.0
        
        # 3D view rotation
        self.rotation_x = 25
        self.rotation_y = -35
        
        # Animation state
        self.anim_faces = None
        self.anim_initial = None
        self.anim_moves = []
        self.anim_idx = 0
        self.anim_playing = False
        self.anim_speed = 500
        self.anim_last_time = 0
        
        # Move animation (smooth rotation)
        self.anim_rotating = False
        self.anim_rotation_progress = 0.0
        self.anim_current_move = None
        self.anim_rotation_duration = 300  # ms per move rotation
        self.anim_reverse_step = False  # True when animating a backward step
        
        # Camera texture
        self.camera_width = 640
        self.camera_height = 480
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        self.running = True
        return True
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def process_frame(self):
        """Process a single camera frame"""
        if not self.cap or not self.running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy() # Keep a clean copy for warping/cropping
        original_h, original_w = frame.shape[:2]
        
        # 1. YOLO Detection
        # Run inference on the frame
        # results = self.yolo_model(clean_frame, verbose=False)
        
        results = self.yolo_model(
            clean_frame, 
            conf=0.25,           # Lower threshold
            iou=0.45,            # Balanced NMS
            verbose=False,
            half=True,           # Use FP16 if GPU supports it
            device=0,            # Explicit GPU
            imgsz=320,           # Match training size
            agnostic_nms=True,   # Faster NMS
        )
        
        # Find best cube (class 0)
        best_box = None
        max_conf = 0
        cube_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Assuming class 0 is cube. Adjust if your model has different classes.
                if cls == 0 and conf > 0.8: 
                    cube_count += 1
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
        
        if cube_count > 1:
             cv2.putText(frame, "WARNING: MULTIPLE CUBES DETECTED!", (50, original_h // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
             cv2.putText(frame, "Please show only one cube.", (50, original_h // 2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
             best_box = None # Prevent processing
        
        self.is_present = False
        self.presence_prob = 0.0
        
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            
            # Ensure crop is within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w, x2)
            y2 = min(original_h, y2)
            
            # Draw YOLO box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Rubik: {max_conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if x2 > x1 and y2 > y1:
                # Crop from CLEAN frame to avoid artifacts
                crop = clean_frame[y1:y2, x1:x2]
                
                # Simplified: Just use the YOLO crop as the "warped" image
                self.is_present = True
                self.presence_prob = max_conf
                
                # Resize to expected size (224x224)
                warped = cv2.resize(crop, (self.img_size, self.img_size))
                
                # Flip warped to match previous logic (un-mirroring if needed)
                warped = cv2.flip(warped, 1)
                
                self.last_warped = warped.copy()
                
                # Draw preview
                preview_size = 100
                warped_small = cv2.resize(warped, (preview_size, preview_size))
                px, py = original_w - preview_size - 10, 10
                frame[py:py+preview_size, px:px+preview_size] = warped_small
                cv2.rectangle(frame, (px, py), (px+preview_size, py+preview_size), (0, 255, 0), 2)

        if not self.is_present:
             self.last_warped = None

        # Draw status
        next_face = self.cube_state.get_next_face()
        if next_face:
            cv2.putText(frame, f"Scan: {next_face.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "COMPLETE!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if not self.is_present:
            cv2.putText(frame, f"No cube ({self.presence_prob:.2f})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Convert BGR to RGBA for DearPyGui
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return frame_rgb
    
    def capture_face(self):
        """Capture current face"""
        next_face = self.cube_state.get_next_face()
        if self.last_warped is not None and next_face:
            lab_grid, _ = extract_face_lab(self.last_warped)
            self.cube_state.set_face_lab(next_face, lab_grid)
            
            center_lab = lab_grid[1][1]
            print(f"✓ Captured {next_face.upper()} - LAB: ({center_lab[0]:.0f}, {center_lab[1]:.0f}, {center_lab[2]:.0f})")
            
            if self.cube_state.is_complete():
                self.finalize_cube()
            return True
        return False
    
    def finalize_cube(self):
        """Resolve colors and get solution"""
        print("\n" + "="*40)
        print("Resolving colors...")
        self.cube_state.resolve_colors()
        
        is_valid, errors = self.cube_state.validate()
        self.validation_errors = errors  # Store for GUI
        if is_valid:
            cube_string = self.cube_state.to_kociemba_string()
            print(f"✓ Valid cube!")
            print(f"Cube string: {cube_string}")
            
            try:
                self.solution = kociemba.solve(cube_string)
                print(f"Solution: {self.solution}")
                print(f"Moves: {len(self.solution.split())}")
            except Exception as e:
                print(f"✗ Kociemba error: {e}")
                self.solution = None
                self.validation_errors = [str(e)]
        else:
            print("✗ Invalid:")
            for e in errors:
                print(f"  {e}")
            self.solution = None
        print("="*40)
    
    def redo_last(self):
        """Redo last captured face"""
        scanned = [c for c in FACE_ORDER if self.cube_state.faces_lab[c] is not None]
        if scanned:
            last_color = scanned[-1]
            self.cube_state.faces_lab[last_color] = None
            self.cube_state.faces[last_color] = None
            if last_color in self.cube_state.center_lab:
                del self.cube_state.center_lab[last_color]
            self.solution = None
            print(f"Redo: {last_color}")
    
    def clear_all(self):
        """Clear all scanned faces"""
        self.cube_state.reset()
        self.solution = None
        self.validation_errors = []
        print("Cleared all")
    
    def init_animation(self):
        """Initialize animation with current solution"""
        if not self.solution:
            return False
        self.anim_moves = self.solution.split()
        self.anim_initial = copy.deepcopy(self.cube_state.faces)
        self.anim_faces = copy.deepcopy(self.cube_state.faces)
        self.anim_idx = 0
        self.anim_playing = False
        self.anim_last_time = time.time()
        return True
    
    def anim_step_forward(self):
        """Step animation forward"""
        if self.anim_idx < len(self.anim_moves):
            move = self.anim_moves[self.anim_idx]
            self.anim_faces = apply_move(self.anim_faces, move)
            self.anim_idx += 1
            return True
        return False
    
    def anim_step_backward(self):
        """Step animation backward"""
        if self.anim_idx > 0:
            self.anim_idx -= 1
            self.anim_faces = copy.deepcopy(self.anim_initial)
            for i in range(self.anim_idx):
                self.anim_faces = apply_move(self.anim_faces, self.anim_moves[i])
            return True
        return False
    
    def anim_reset(self):
        """Reset animation to start"""
        self.anim_idx = 0
        self.anim_faces = copy.deepcopy(self.anim_initial)
        self.anim_playing = False


# ============================================================================
# DEARPYGUI APPLICATION
# ============================================================================

def run_app():
    """Run the DearPyGui application"""
    # Load model
    # model, device = load_model("rubik_model_pytorch.pt")
    app = RubikScannerApp()
    
    if not app.start_camera():
        return
    
    # Create DPG context
    dpg.create_context()
    
    # Create textures
    cam_width, cam_height = 640, 480
    with dpg.texture_registry():
        dpg.add_raw_texture(cam_width, cam_height, 
                           np.zeros((cam_height, cam_width, 4), dtype=np.float32).flatten(),
                           format=dpg.mvFormat_Float_rgba, tag="camera_texture")
    
    # Theme
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (35, 35, 45))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (45, 45, 55))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 65, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 85, 100))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 105, 120))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 42, 50))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (100, 180, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
    dpg.bind_theme(global_theme)
    
    # Callbacks
    def on_capture():
        app.capture_face()
        update_ui()
    
    def on_redo():
        app.redo_last()
        update_ui()
    
    def on_clear():
        app.clear_all()
        update_ui()
        # Clear the flat cube preview canvas
        dpg.delete_item("cube_preview", children_only=True)
        dpg.draw_rectangle((0, 0), (280, 220), fill=(25, 25, 30, 255), parent="cube_preview")
    
    def on_3d_view():
        if app.cube_state.get_scanned_count() > 0:
            if app.cube_state.faces[FACE_ORDER[0]] is None:
                app.cube_state.resolve_colors()
            dpg.configure_item("3d_window", show=True)
            draw_3d_cube()
    
    def on_animate():
        if app.solution and app.init_animation():
            dpg.configure_item("anim_window", show=True)
            update_anim_ui()
            draw_anim_cube()
    
    def draw_3d_cube():
        if app.cube_state.faces[FACE_ORDER[0]] is not None:
            draw_cube_on_canvas("3d_canvas", app.cube_state.faces, 
                               app.rotation_x, app.rotation_y, 550, 450)
    
    def draw_anim_cube():
        if app.anim_faces:
            if app.anim_rotating and app.anim_current_move:
                # Get rotation angle based on progress (0-90 or 0-180 for double moves)
                move = app.anim_current_move
                if '2' in move:
                    target_angle = 180
                else:
                    target_angle = 90
                
                # Get direction from MOVE_INFO
                if move in MOVE_INFO:
                    _, _, direction = MOVE_INFO[move]
                    angle = target_angle * app.anim_rotation_progress * direction
                    draw_animated_cube("anim_canvas", app.anim_faces,
                                      app.rotation_x, app.rotation_y, 600, 400,
                                      rotating_face=move, rotation_angle=angle)
                else:
                    draw_cube_on_canvas("anim_canvas", app.anim_faces,
                                       app.rotation_x, app.rotation_y, 600, 400)
            else:
                draw_cube_on_canvas("anim_canvas", app.anim_faces,
                                   app.rotation_x, app.rotation_y, 600, 400)
    
    def update_ui():
        # Update progress
        count = app.cube_state.get_scanned_count()
        dpg.set_value("progress_bar", count / 6)
        dpg.set_value("progress_text", f"{count}/6 faces scanned")
        
        # Update next face
        next_face = app.cube_state.get_next_face()
        if next_face:
            dpg.set_value("next_face_text", f"Next: {next_face.upper()}")
        else:
            dpg.set_value("next_face_text", "All faces captured!")
        
        # Update face status indicators
        for i, face in enumerate(FACE_ORDER):
            is_done = app.cube_state.faces_lab[face] is not None
            color = (80, 220, 120) if is_done else (100, 100, 110)
            dpg.configure_item(f"face_status_{i}", color=color)
        
        # Update solution or show errors
        if app.solution:
            moves = app.solution.split()
            dpg.set_value("solution_text", f"Solution: {len(moves)} moves")
            dpg.configure_item("solution_text", color=(80, 220, 120))
            dpg.set_value("solution_moves", app.solution)
        elif app.validation_errors and app.cube_state.get_scanned_count() == 6:
            # Show validation errors in red
            error_msg = "Invalid: " + "; ".join(app.validation_errors[:2])  # Show first 2 errors
            dpg.set_value("solution_text", error_msg)
            dpg.configure_item("solution_text", color=(255, 80, 80))
            dpg.set_value("solution_moves", "")
        else:
            dpg.set_value("solution_text", "")
            dpg.set_value("solution_moves", "")
    
    def update_anim_ui():
        if app.anim_moves:
            progress = app.anim_idx / len(app.anim_moves)
            dpg.set_value("anim_progress", progress)
            dpg.set_value("anim_move_text", f"Move {app.anim_idx} / {len(app.anim_moves)}")
            
            if app.anim_idx < len(app.anim_moves):
                dpg.set_value("anim_next_move", f"Next: {app.anim_moves[app.anim_idx]}")
            else:
                dpg.set_value("anim_next_move", "SOLVED!")
    
    # 3D view keyboard rotation helper
    def adjust_view(delta_yaw=0.0, delta_pitch=0.0):
        """Nudge view rotation using keyboard (arrow keys)."""
        app.rotation_y += delta_yaw
        app.rotation_x = max(-90, min(90, app.rotation_x + delta_pitch))

        if dpg.is_item_shown("3d_window"):
            draw_3d_cube()
        if dpg.is_item_shown("anim_window"):
            draw_anim_cube()
    
    # Animation controls
    def anim_play_pause():
        if not app.anim_rotating:  # Don't toggle if mid-rotation
            app.anim_playing = not app.anim_playing
            dpg.set_value("anim_play_btn", "Pause" if app.anim_playing else "Play")
            app.anim_last_time = time.time()
    
    def anim_step_fwd():
        # Manual step with animation
        if not app.anim_rotating and app.anim_idx < len(app.anim_moves):
            app.anim_rotating = True
            app.anim_current_move = app.anim_moves[app.anim_idx]
            app.anim_rotation_progress = 0.0
            app.anim_last_time = time.time()
            # Don't increment anim_idx here - it's done when rotation completes
    
    def anim_step_back():
        # Manual step backward with animation
        if not app.anim_rotating and app.anim_idx > 0:
            # Get the move we're undoing (previous move)
            prev_move = app.anim_moves[app.anim_idx - 1]
            # Convert to reverse move for animation direction
            if "'" in prev_move:
                reverse_move = prev_move.replace("'", "")
            elif "2" in prev_move:
                reverse_move = prev_move  # 180 is same both ways
            else:
                reverse_move = prev_move + "'"
            
            # First, go back one step in the state (without animation)
            app.anim_idx -= 1
            app.anim_faces = copy.deepcopy(app.anim_initial)
            for i in range(app.anim_idx):
                app.anim_faces = apply_move(app.anim_faces, app.anim_moves[i])
            
            # Now animate the reverse move visually
            app.anim_rotating = True
            app.anim_current_move = reverse_move
            app.anim_rotation_progress = 0.0
            app.anim_last_time = time.time()
            app.anim_reverse_step = True  # Flag to skip state change after animation
            update_anim_ui()
    
    def anim_reset():
        app.anim_reset()
        app.anim_rotating = False
        app.anim_current_move = None
        app.anim_rotation_progress = 0.0
        dpg.set_value("anim_play_btn", "Play")
        update_anim_ui()
        draw_anim_cube()
    
    def anim_speed_change(sender, value):
        # Speed slider: 10 (slow) to 100 (fast)
        # Map to duration: 500ms at slow end to 100ms at fast end
        app.anim_rotation_duration = int(600 - value * 5)
    
    # Keyboard handler
    def on_key_press(sender, app_data):
        key = app_data
        if key == dpg.mvKey_Spacebar:
            on_capture()
        elif key == dpg.mvKey_R:
            on_redo()
        elif key == dpg.mvKey_C:
            on_clear()
        elif key == dpg.mvKey_V:
            on_3d_view()
        elif key == dpg.mvKey_A:
            on_animate()
        elif key == dpg.mvKey_Left:
            adjust_view(delta_yaw=-6)
        elif key == dpg.mvKey_Right:
            adjust_view(delta_yaw=6)
        elif key == dpg.mvKey_Up:
            adjust_view(delta_pitch=-6)
        elif key == dpg.mvKey_Down:
            adjust_view(delta_pitch=6)
        elif key == dpg.mvKey_Q:
            dpg.stop_dearpygui()
    
    # Register handlers
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press)
    
    # ============ MAIN WINDOW ============
    with dpg.window(label="Rubik's Cube Scanner", tag="main_window", width=1200, height=700):
        with dpg.group(horizontal=True):
            # Left panel - Guide
            with dpg.child_window(width=200, height=-1, border=False):
                dpg.add_text("RUBIK SCANNER", color=(100, 200, 255))
                dpg.add_separator()
                
                dpg.add_spacer(height=10)
                dpg.add_text("Progress", color=(150, 150, 160))
                dpg.add_progress_bar(tag="progress_bar", default_value=0, width=-1)
                dpg.add_text("0/6 faces scanned", tag="progress_text", color=(180, 180, 190))
                
                dpg.add_spacer(height=15)
                dpg.add_text("", tag="next_face_text", color=(255, 200, 100))
                
                dpg.add_spacer(height=15)
                dpg.add_separator()
                dpg.add_text("Scan Order", color=(150, 150, 160))
                
                face_info = [
                    ("WHITE", (255, 255, 255), "tilt up"),
                    ("YELLOW", (255, 255, 0), "tilt down"),
                    ("GREEN", (0, 200, 0), "front"),
                    ("ORANGE", (255, 140, 0), "rotate"),
                    ("BLUE", (0, 100, 255), "rotate"),
                    ("RED", (255, 0, 0), "rotate"),
                ]
                for i, (name, color, action) in enumerate(face_info):
                    with dpg.group(horizontal=True):
                        dpg.add_text("●", tag=f"face_status_{i}", color=(100, 100, 110))
                        dpg.add_text(name, color=color)
                        dpg.add_text(f"({action})", color=(100, 100, 110))
                
                dpg.add_spacer(height=15)
                dpg.add_separator()
                dpg.add_text("Controls", color=(150, 150, 160))
                dpg.add_text("SPACE - Capture", color=(120, 120, 130))
                dpg.add_text("R - Redo last", color=(120, 120, 130))
                dpg.add_text("C - Clear all", color=(120, 120, 130))
                dpg.add_text("V - 3D View", color=(120, 120, 130))
                dpg.add_text("A - Animate", color=(120, 120, 130))
                dpg.add_text("Arrow Keys - Rotate view", color=(120, 120, 130))
                dpg.add_text("Q - Quit", color=(120, 120, 130))
                
                dpg.add_spacer(height=15)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Capture", callback=on_capture, width=90)
                    dpg.add_button(label="Redo", callback=on_redo, width=90)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Clear", callback=on_clear, width=90)
                    dpg.add_button(label="3D View", callback=on_3d_view, width=90)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Animate", callback=on_animate, width=90)
                    dpg.add_button(label="Quit", callback=lambda: dpg.stop_dearpygui(), width=90)
            
            # Center - Camera feed
            with dpg.child_window(width=660, height=-1, border=False):
                dpg.add_text("Camera Feed", color=(150, 150, 160))
                dpg.add_image("camera_texture", width=640, height=480)
            
            # Right panel - Cube state
            with dpg.child_window(width=-1, height=-1, border=False):
                dpg.add_text("CUBE STATE", color=(100, 200, 255))
                dpg.add_separator()
                
                with dpg.drawlist(width=280, height=280, tag="cube_preview"):
                    pass
                
                dpg.add_spacer(height=10)
                dpg.add_text("", tag="solution_text", color=(80, 220, 120))
                dpg.add_text("", tag="solution_moves", color=(180, 180, 190), wrap=280)
    
    # ============ 3D VIEWER WINDOW ============
    with dpg.window(label="3D Cube Viewer", tag="3d_window", width=600, height=550, 
                    show=False, pos=(300, 75)):
        dpg.add_text("Use arrow keys to rotate the cube", color=(150, 150, 160))
        dpg.add_separator()
        with dpg.drawlist(width=550, height=450, tag="3d_canvas"):
            pass
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset View", callback=lambda: (
                setattr(app, 'rotation_x', 25), 
                setattr(app, 'rotation_y', -35), 
                draw_3d_cube()
            ))
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("3d_window", show=False))
    
    # ============ ANIMATION WINDOW ============
    with dpg.window(label="Solution Animator", tag="anim_window", width=650, height=600,
                    show=False, pos=(275, 50)):
        dpg.add_text("Use arrow keys to rotate view | Use controls below", color=(150, 150, 160))
        dpg.add_separator()
        
        with dpg.drawlist(width=600, height=400, tag="anim_canvas"):
            pass
        
        dpg.add_separator()
        dpg.add_progress_bar(tag="anim_progress", default_value=0, width=-1)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Move 0 / 0", tag="anim_move_text")
            dpg.add_spacer(width=30)
            dpg.add_text("", tag="anim_next_move", color=(100, 200, 255))
        
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play", tag="anim_play_btn", callback=anim_play_pause, width=70)
            dpg.add_button(label="<< Prev", callback=anim_step_back, width=70)
            dpg.add_button(label="Next >>", callback=anim_step_fwd, width=70)
            dpg.add_button(label="Reset", callback=anim_reset, width=70)
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("anim_window", show=False), width=70)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Speed:", color=(150, 150, 160))
            dpg.add_slider_int(default_value=60, min_value=10, max_value=100, 
                              width=200, callback=anim_speed_change)
    
    # Create viewport
    dpg.create_viewport(title="Rubik's Cube Scanner", width=1220, height=720)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # Main loop
    try:
        while dpg.is_dearpygui_running():
            # Process camera frame
            frame = app.process_frame()
            if frame is not None:
                # Update texture
                frame_float = frame.astype(np.float32) / 255.0
                dpg.set_value("camera_texture", frame_float.flatten())
            
            # Update cube preview if faces scanned (use flattened view)
            if app.cube_state.get_scanned_count() > 0:
                if app.cube_state.faces[FACE_ORDER[0]] is None:
                    app.cube_state.resolve_colors()
                draw_flat_cube_preview("cube_preview", app.cube_state, 280, 220)
            
            # Handle animation playback with smooth transitions
            if dpg.is_item_shown("anim_window"):
                current_time = time.time()
                
                if app.anim_rotating:
                    # Currently animating a move rotation
                    elapsed = (current_time - app.anim_last_time) * 1000
                    app.anim_rotation_progress = min(1.0, elapsed / app.anim_rotation_duration)
                    
                    if app.anim_rotation_progress >= 1.0:
                        # Rotation complete
                        if app.anim_reverse_step:
                            # Backward step - state already updated, just end animation
                            app.anim_reverse_step = False
                        else:
                            # Forward step - apply the move and increment index
                            app.anim_faces = apply_move(app.anim_faces, app.anim_current_move)
                            app.anim_idx += 1
                        
                        app.anim_rotating = False
                        app.anim_current_move = None
                        app.anim_rotation_progress = 0.0
                        update_anim_ui()
                        
                        # Check if we're done (only for forward playing)
                        if app.anim_idx >= len(app.anim_moves):
                            app.anim_playing = False
                            dpg.set_value("anim_play_btn", "Play")
                    
                    draw_anim_cube()
                    
                elif app.anim_playing:
                    # Ready to start next move animation
                    if app.anim_idx < len(app.anim_moves):
                        app.anim_rotating = True
                        app.anim_current_move = app.anim_moves[app.anim_idx]
                        app.anim_rotation_progress = 0.0
                        app.anim_last_time = current_time
            
            dpg.render_dearpygui_frame()
    finally:
        app.stop_camera()
        dpg.destroy_context()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_app()
