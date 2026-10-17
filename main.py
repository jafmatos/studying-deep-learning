from src.config import ROOT_DIR
from pathlib import Path
from PIL import Image

IMAGE_PATH = ROOT_DIR / "data" / "divina-misericordia.png"

image = Image.open(IMAGE_PATH)

print(f"Formato: {image.format}")
print(f"Modo: {image.mode}")
print(f"Tamanho (W, H): {image.size}")
