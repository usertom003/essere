from typing import Dict, Any

TEST_CONFIG: Dict[str, Any] = {
    'video': {
        'width': 640,
        'height': 480,
        'fps': 30
    },
    'analysis': {
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'max_faces': 1
    },
    'performance': {
        'max_memory_usage': 1000,  # MB
        'max_cpu_usage': 80,  # %
        'target_fps': 15
    },
    'paths': {
        'models': 'models',
        'logs': 'logs',
        'data': 'data',
        'temp': 'temp'
    }
}

MOCK_DATA_CONFIG = {
    'frame_dimensions': (480, 640, 3),
    'sequence_length': 100,
    'feature_dimensions': 99,
    'batch_size': 32
} 