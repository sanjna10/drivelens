from pathlib import Path

# Repository root: .../drivelens 4/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "driving_dataset_2k"
ANNOTATION_FILE = DATA_DIR / "_annotations.coco.json"

CLIP_STORE_DIR = PROJECT_ROOT / "clip_store"
FAISS_INDEX_FILE = CLIP_STORE_DIR / "driving_clip.index"
IMAGE_METADATA_FILE = CLIP_STORE_DIR / "image_metadata.csv"

DETECTOR_WEIGHTS_FILE = PROJECT_ROOT / "retinanet_resnet50.weights.h5"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

DETECTION_IMAGE_SIZE = (640, 640)
