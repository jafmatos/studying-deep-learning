from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "Train"
TEST_DATA_DIR = DATA_DIR / "Test"
VALIDATION_DATA_DIR = DATA_DIR / "Validation"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

LEARNING_RATE = 0.001
NUM_EPOCHS = 20
