import numpy
import torch
from PIL import Image, ImageOps
from PIL.Image import Transpose

from deep_learning.config import IMAGE_SIZE


def train_transform(image: Image.Image) -> torch.Tensor:
    if numpy.random.rand() < 0.5:
        angle = float(numpy.random.uniform(-15.0, 15.0))
        image = image.rotate(angle=angle)

    if numpy.random.rand() < 0.5:
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT)

    image = ImageOps.fit(image, size=IMAGE_SIZE, method=Image.Resampling.BILINEAR)

    tensor_numpy = numpy.array(image, dtype=numpy.float32) / 255
    tensor_numpy = tensor_numpy.transpose(2, 0, 1)

    return torch.from_numpy(tensor_numpy)


def test_transform(image: Image.Image) -> torch.Tensor:
    image = ImageOps.fit(image, size=IMAGE_SIZE, method=Image.Resampling.BILINEAR)

    tensor_numpy = numpy.array(image, dtype=numpy.float32) / 255
    tensor_numpy = tensor_numpy.transpose(2, 0, 1)

    return torch.from_numpy(tensor_numpy)


def validation_transform(image: Image.Image) -> torch.Tensor:
    image = ImageOps.fit(image, size=IMAGE_SIZE, method=Image.Resampling.BILINEAR)

    tensor_numpy = numpy.array(image, dtype=numpy.float32) / 255
    tensor_numpy = tensor_numpy.transpose(2, 0, 1)

    return torch.from_numpy(tensor_numpy)
