import cv2
import numpy as np
from insightface.app import FaceAnalysis

cuda_provider_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}

class InsightFaceEngine:
    def __init__(self, model_name="buffalo_l", root_dir="."):
        """
        Initializes the InsightFace analysis app.
        :param model_name: 'buffalo_l' or 'buffalo_s'
        :param root_dir: Directory containing the 'models' folder (usually current dir '.')
        """
        # FaceAnalysis automatically looks for models inside {root_dir}/models/{model_name}
        self.app = FaceAnalysis(
            name=model_name, 
            root=root_dir, 
            providers=[('CUDAExecutionProvider', cuda_provider_options), 
                       'CPUExecutionProvider'
            ],
            allowed_modules=['detection', 'recognition'] # <--- Added this line
        )
        
        # Prepare the model (ctx_id=0 for GPU, -1 for CPU)
        # det_size=(640, 640) is standard; (320, 320) is faster but less accurate for small faces
        try:
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"InsightFace {model_name} loaded on GPU.")
        except Exception as e:
            print(f"GPU initialization failed ({e}), falling back to CPU...")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Just CPU
        # self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def process_frame(self, frame_bgr, threshold=0.5):
        """
        Detects and embeds faces in a single pass.
        Returns a list of dicts compatible with the existing GUI app.
        """
        # InsightFace expects BGR (OpenCV default)
        faces = self.app.get(frame_bgr)
        
        results = []
        for face in faces:
            if face.det_score < threshold:
                continue
            # Convert float bbox to int list [x1, y1, x2, y2]
            bbox = face.bbox.astype(int).tolist()
            
            results.append({
                'bbox': bbox,
                'confidence': float(face.det_score),
                'landmarks': face.kps,       # 5 Keypoints used for alignment
                'embedding': face.embedding  # 512D Vector (already aligned and normalized)
            })
            
        return results