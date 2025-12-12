"""
Rubik's Cube face reading and color detection module
Handles LAB color extraction, clustering-based color resolution, and kociemba string generation
"""

import numpy as np
import cv2
from skimage import color


FACE_ORDER = ['white', 'yellow', 'green', 'orange', 'blue', 'red']

# Map face colors to kociemba notation
FACE_TO_KOCIEMBA = {
    'white': 'U',
    'yellow': 'D',
    'green': 'F',
    'red': 'R',
    'blue': 'B',
    'orange': 'L',
}

# Color name translation between our scheme and pycuber defaults
OUR_TO_PY_COLOR = {
    'white': 'yellow',  # our U is white, pycuber U is yellow by default
    'yellow': 'white',
    'green': 'green',
    'blue': 'blue',
    'red': 'orange',
    'orange': 'red',
    'unknown': 'unknown',
}

PY_TO_OUR_COLOR = {v: k for k, v in OUR_TO_PY_COLOR.items()}

FACE_TO_PY_FACE = {
    'white': 'U',
    'red': 'R',
    'green': 'F',
    'yellow': 'D',
    'orange': 'L',
    'blue': 'B',
}

PY_FACE_TO_FACE = {v: k for k, v in FACE_TO_PY_FACE.items()}

# Coordinate frame helpers for cube rotations (x: left→right, y: down→up, z: front→back)
FACE_AXES = {
    'white': {
        'normal': (0, 1, 0),
        'row_axis': (0, 0, 1),
        'col_axis': (1, 0, 0),
    },
    'yellow': {
        'normal': (0, -1, 0),
        'row_axis': (0, 0, -1),
        'col_axis': (1, 0, 0),
    },
    'green': {
        'normal': (0, 0, 1),
        'row_axis': (0, -1, 0),
        'col_axis': (1, 0, 0),
    },
    'blue': {
        'normal': (0, 0, -1),
        'row_axis': (0, -1, 0),
        'col_axis': (-1, 0, 0),
    },
    'red': {
        'normal': (1, 0, 0),
        'row_axis': (0, -1, 0),
        'col_axis': (0, 0, -1),
    },
    'orange': {
        'normal': (-1, 0, 0),
        'row_axis': (0, -1, 0),
        'col_axis': (0, 0, 1),
    },
}

FACE_COORDS = {}
COORD_TO_FACE = {}
for face_name, axes in FACE_AXES.items():
    normal = axes['normal']
    row_axis = axes['row_axis']
    col_axis = axes['col_axis']
    for row in range(3):
        for col in range(3):
            coord = tuple(
                normal[i]
                + (row - 1) * row_axis[i]
                + (col - 1) * col_axis[i]
                for i in range(3)
            )
            FACE_COORDS[(face_name, row, col)] = coord
            COORD_TO_FACE[coord] = (face_name, row, col)

AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

MOVE_SPECS = {
    'U': ('y', 1, -90),
    'D': ('y', -1, 90),
    'F': ('z', 1, -90),
    'B': ('z', -1, 90),
    'R': ('x', 1, -90),
    'L': ('x', -1, 90),
}

class CubeState:
    """Manages the state of a Rubik's cube"""
    
    def __init__(self):
        self.faces = {face: None for face in FACE_ORDER}
        self.faces_lab = {face: None for face in FACE_ORDER}
        self.center_lab = {}
        
    def reset(self):
        """Reset cube state"""
        self.faces = {face: None for face in FACE_ORDER}
        self.faces_lab = {face: None for face in FACE_ORDER}
        self.center_lab = {}
    
    def get_scanned_count(self):
        """Count how many faces have been scanned"""
        return sum(1 for face in FACE_ORDER if self.faces_lab[face] is not None)
    
    def get_next_face(self):
        """Get the next face to scan"""
        for face in FACE_ORDER:
            if self.faces_lab[face] is None:
                return face
        return None
    
    def is_complete(self):
        """Check if all faces have been scanned"""
        return all(self.faces_lab[face] is not None for face in FACE_ORDER)
    
    def set_face_lab(self, face, lab_array):
        """Set LAB values for a face (3x3x3 array where last dim is L, a, b)"""
        self.faces_lab[face] = np.copy(lab_array)
        # Store center color
        center_lab = lab_array[1, 1]
        self.center_lab[face] = center_lab
    
    def resolve_colors(self):
        """
        Resolve colors using DeltaE2000 color distance.
        The 6 center stickers define the 6 color clusters.
        All 54 stickers are assigned using DeltaE2000 perceptual color difference.
        Uses proper LAB format conversion for accurate color science.
        """
        # Collect all sticker LAB values with their physical positions
        all_stickers = []
        sticker_positions = []
        
        for face in FACE_ORDER:
            if self.faces_lab[face] is None:
                continue
            lab_face = self.faces_lab[face]
            for row in range(3):
                for col in range(3):
                    sticker_lab = lab_face[row, col]
                    all_stickers.append(sticker_lab)
                    sticker_positions.append((face, row, col))
        
        if not all_stickers:
            return
        
        all_stickers = np.array(all_stickers)
        
        # Build centroids from the 6 center colors
        centroids = []
        centroid_colors = []
        for face in FACE_ORDER:
            if face in self.center_lab:
                centroids.append(self.center_lab[face])
                centroid_colors.append(face)
        
        if len(centroids) != 6:
            return
        
        centroids = np.array(centroids)
        
        # DON'T smooth centroids - trust the center stickers directly
        # The center stickers define the absolute color identity
        
        # Convert OpenCV LAB to standard LAB format for DeltaE2000
        # OpenCV: L=[0,255], a=[0,255], b=[0,255]
        # Standard: L=[0,100], a=[-128,127], b=[-128,127]
        all_stickers_std = opencv_lab_to_standard_lab(all_stickers)
        centroids_std = opencv_lab_to_standard_lab(centroids)
        
        # Vectorized DeltaE2000 computation for efficiency
        # Compute all distances at once using skimage's deltaE_ciede2000
        assignments = []
        for sticker_lab in all_stickers_std:
            # Reshape for broadcasting: sticker is (3,), centroids is (6,3)
            # deltaE_ciede2000 expects (1,3) vs (6,3)
            sticker_reshaped = sticker_lab.reshape(1, 1, 3)
            centroids_reshaped = centroids_std.reshape(1, -1, 3)
            
            # Compute DeltaE2000 to all centroids at once
            deltas = color.deltaE_ciede2000(
                sticker_reshaped,
                centroids_reshaped
            )[0]  # Result shape: (6,)
            
            # Assign to closest centroid
            best_idx = np.argmin(deltas)
            assignments.append(best_idx)
        
        # Initialize face arrays - each position on cube stores its COLOR
        self.faces = {face: np.full((3, 3), 'unknown', dtype=object) for face in FACE_ORDER}
        
        # Fill in colors based on assignments
        for i, (face, row, col) in enumerate(sticker_positions):
            assigned_color = centroid_colors[assignments[i]]
            self.faces[face][row, col] = assigned_color
    
    def validate(self):
        """
        Validate the cube state.
        Returns (is_valid, error_list)
        """
        errors = []
        
        if not all(self.faces[f] is not None for f in FACE_ORDER):
            errors.append("Not all faces scanned")
            return False, errors
        
        # Count each color
        color_counts = {face: 0 for face in FACE_ORDER}
        for face in FACE_ORDER:
            for row in range(3):
                for col in range(3):
                    color = self.faces[face][row, col]
                    if color in color_counts:
                        color_counts[color] += 1
                    else:
                        errors.append(f"Unknown color: {color}")
        
        # Each color should appear exactly 9 times
        for color, count in color_counts.items():
            if count != 9:
                errors.append(f"{color}: {count}/9 stickers")
        
        # Verify centers match their face identity
        # In a valid scan, white face center should be white, etc.
        for face_name in FACE_ORDER:
            center_color = self.faces[face_name][1, 1]
            if center_color != face_name:
                errors.append(f"{face_name} face has {center_color} center (should be {face_name})")
        
        if errors:
            return False, errors
        
        return True, []
    
    def to_kociemba_string(self):
        """
        Convert cube state to kociemba string format.
        String format: 54 characters representing 6 faces in order U R F D L B
        Each face: 9 characters for top-left to bottom-right (row by row)
        
        Note: self.faces[face_name][row, col] contains the COLOR at that position,
        not which face it belongs to.
        
        Face orientations for kociemba (looking at each face directly):
        - U (white/top): when looking down, F should be toward you
        - R (red/right): when looking at right face, U should be on top
        - F (green/front): when looking at front, U should be on top
        - D (yellow/bottom): when looking up from below, F should be toward you  
        - L (orange/left): when looking at left face, U should be on top
        - B (blue/back): when looking at back, U should be on top
        """
        face_sequence = ['white', 'red', 'green', 'yellow', 'orange', 'blue']
        parts = []

        for face_name in face_sequence:
            face_array = self.faces[face_name]
            for row in range(3):
                for col in range(3):
                    sticker_color = face_array[row, col]
                    if sticker_color not in FACE_TO_KOCIEMBA:
                        raise ValueError(f"Unknown sticker color '{sticker_color}' on face '{face_name}'")
                    parts.append(FACE_TO_KOCIEMBA[sticker_color])

        return ''.join(parts)


def normalize_illumination(img):
    """
    Apply gentle illumination normalization using Gray-World assumption.
    This stabilizes colors across different lighting conditions:
    - Yellow lights, blue daylight, shadows, monitors, LEDs
    Makes overall RGB levels balanced to neutral white.
    """
    # Convert to float32 for precision
    img_float = img.astype(np.float32)
    
    # Gray-World normalization: scale each channel so its mean = 128
    # But limit scaling factor to avoid extreme adjustments
    for channel in range(3):
        channel_mean = np.mean(img_float[:, :, channel])
        if channel_mean > 0:
            scale = 128.0 / channel_mean
            # Limit scale to reasonable range [0.7, 1.5] to avoid over-correction
            scale = np.clip(scale, 0.7, 1.5)
            img_float[:, :, channel] = np.clip(img_float[:, :, channel] * scale, 0, 255)
    
    return img_float.astype(np.uint8)


def opencv_lab_to_standard_lab(lab_opencv):
    """
    Convert OpenCV LAB format to standard LAB format for DeltaE2000.
    OpenCV: L=[0,255], a=[0,255], b=[0,255]
    Standard: L=[0,100], a=[-128,127], b=[-128,127]
    """
    lab_standard = lab_opencv.copy()
    lab_standard[..., 0] = lab_opencv[..., 0] * 100.0 / 255.0  # L: 0-255 -> 0-100
    lab_standard[..., 1] = lab_opencv[..., 1] - 128.0          # a: 0-255 -> -128 to 127
    lab_standard[..., 2] = lab_opencv[..., 2] - 128.0          # b: 0-255 -> -128 to 127
    return lab_standard


def extract_face_lab(warped_face):
    """
    Extract LAB values from a warped cube face image (224x224) with:
    1. Direct LAB color space conversion (no normalization - LAB is inherently illumination-invariant)
    2. Circular sampling around sticker centers
    3. Median LAB computation for robustness
    
    Returns (stickers_lab, warped_face_normalized)
    stickers_lab: 3x3x3 array where last dimension is [L, a, b]
    """
    h, w = warped_face.shape[:2]
    cell_size = h // 3
    
    # Convert directly to LAB color space without normalization
    # LAB is inherently more invariant to illumination than RGB
    face_lab = cv2.cvtColor(warped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Extract sticker colors using circular sampling
    stickers_lab = np.zeros((3, 3, 3), dtype=np.float32)
    
    for row in range(3):
        for col in range(3):
            # Find center of this cell
            y_center = (row * cell_size) + (cell_size // 2)
            x_center = (col * cell_size) + (cell_size // 2)
            
            # Use circular sampling region (~40% of cell radius)
            # Larger radius for more stable samples while avoiding edges
            radius = int(cell_size * 0.40)
            
            # Create circular mask
            y_indices, x_indices = np.ogrid[:h, :w]
            mask = (y_indices - y_center)**2 + (x_indices - x_center)**2 <= radius**2
            
            # Extract pixels within circular region
            circular_region = face_lab[mask]
            
            if circular_region.size > 0:
                # Use median for robustness against:
                # - shadows, imperfect warps, dust, wear on stickers
                stickers_lab[row, col] = np.median(circular_region, axis=0)
            else:
                # Fallback: use center pixel
                stickers_lab[row, col] = face_lab[y_center, x_center]
    
    return stickers_lab, warped_face


def apply_move(faces, move):
    """
    Apply a Rubik's cube move to the face state.
    Returns a new copy with the move applied.
    Manual implementation aligned with physical cube orientation
    (white top, yellow bottom, green front, blue back, red right, orange left).
    """
    import copy
    faces = copy.deepcopy(faces)

    def rotate_face_cw(face_array):
        return np.rot90(face_array, k=-1)

    def rotate_face_ccw(face_array):
        return np.rot90(face_array, k=1)

    def reverse(arr):
        return arr[::-1]

    if move == 'U':
        faces['white'] = rotate_face_cw(faces['white'])
        temp = faces['green'][0].copy()
        faces['green'][0] = faces['red'][0].copy()
        faces['red'][0] = faces['blue'][0].copy()
        faces['blue'][0] = faces['orange'][0].copy()
        faces['orange'][0] = temp

    elif move == "U'":
        faces['white'] = rotate_face_ccw(faces['white'])
        temp = faces['green'][0].copy()
        faces['green'][0] = faces['orange'][0].copy()
        faces['orange'][0] = faces['blue'][0].copy()
        faces['blue'][0] = faces['red'][0].copy()
        faces['red'][0] = temp

    elif move == 'D':
        faces['yellow'] = rotate_face_cw(faces['yellow'])
        temp = faces['green'][2].copy()
        faces['green'][2] = faces['orange'][2].copy()
        faces['orange'][2] = faces['blue'][2].copy()
        faces['blue'][2] = faces['red'][2].copy()
        faces['red'][2] = temp

    elif move == "D'":
        faces['yellow'] = rotate_face_ccw(faces['yellow'])
        temp = faces['green'][2].copy()
        faces['green'][2] = faces['red'][2].copy()
        faces['red'][2] = faces['blue'][2].copy()
        faces['blue'][2] = faces['orange'][2].copy()
        faces['orange'][2] = temp

    elif move == 'F':
        faces['green'] = rotate_face_cw(faces['green'])
        temp = faces['white'][2].copy()
        faces['white'][2] = reverse(faces['orange'][:, 2]).copy()
        faces['orange'][:, 2] = faces['yellow'][0].copy()
        faces['yellow'][0] = reverse(faces['red'][:, 0]).copy()
        faces['red'][:, 0] = temp

    elif move == "F'":
        faces['green'] = rotate_face_ccw(faces['green'])
        temp = faces['white'][2].copy()
        faces['white'][2] = faces['red'][:, 0].copy()
        faces['red'][:, 0] = reverse(faces['yellow'][0]).copy()
        faces['yellow'][0] = faces['orange'][:, 2].copy()
        faces['orange'][:, 2] = reverse(temp).copy()

    elif move == 'B':
        faces['blue'] = rotate_face_cw(faces['blue'])
        temp = faces['white'][0].copy()
        faces['white'][0] = faces['red'][:, 2].copy()
        faces['red'][:, 2] = reverse(faces['yellow'][2]).copy()
        faces['yellow'][2] = faces['orange'][:, 0].copy()
        faces['orange'][:, 0] = reverse(temp).copy()

    elif move == "B'":
        faces['blue'] = rotate_face_ccw(faces['blue'])
        temp = faces['white'][0].copy()
        faces['white'][0] = reverse(faces['orange'][:, 0]).copy()
        faces['orange'][:, 0] = faces['yellow'][2].copy()
        faces['yellow'][2] = reverse(faces['red'][:, 2]).copy()
        faces['red'][:, 2] = temp

    elif move == 'R':
        faces['red'] = rotate_face_cw(faces['red'])
        temp = faces['white'][:, 2].copy()
        faces['white'][:, 2] = faces['green'][:, 2]
        faces['green'][:, 2] = faces['yellow'][:, 2]
        faces['yellow'][:, 2] = reverse(faces['blue'][:, 0])
        faces['blue'][:, 0] = reverse(temp)

    elif move == "R'":
        faces['red'] = rotate_face_ccw(faces['red'])
        temp = faces['white'][:, 2].copy()
        faces['white'][:, 2] = reverse(faces['blue'][:, 0]).copy()
        faces['blue'][:, 0] = reverse(faces['yellow'][:, 2]).copy()
        faces['yellow'][:, 2] = faces['green'][:, 2].copy()
        faces['green'][:, 2] = temp

    elif move == 'L':
        faces['orange'] = rotate_face_cw(faces['orange'])
        temp = faces['white'][:, 0].copy()
        faces['white'][:, 0] = reverse(faces['blue'][:, 2]).copy()
        faces['blue'][:, 2] = reverse(faces['yellow'][:, 0]).copy()
        faces['yellow'][:, 0] = faces['green'][:, 0].copy()
        faces['green'][:, 0] = temp

    elif move == "L'":
        faces['orange'] = rotate_face_ccw(faces['orange'])
        temp = faces['white'][:, 0].copy()
        faces['white'][:, 0] = faces['green'][:, 0].copy()
        faces['green'][:, 0] = faces['yellow'][:, 0].copy()
        faces['yellow'][:, 0] = reverse(faces['blue'][:, 2]).copy()
        faces['blue'][:, 2] = reverse(temp).copy()

    elif move.endswith('2'):
        base = move[0]
        faces = apply_move(faces, base)
        faces = apply_move(faces, base)

    elif move.endswith("'"):
        base = move[0]
        for _ in range(3):
            faces = apply_move(faces, base)

    else:
        raise ValueError(f"Unsupported move: {move}")

    return faces
