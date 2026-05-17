from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple

import torch
import imageio.v3 as iio
from PIL import Image


PathLike = Union[str, Path]


def _to_uint8_image(
    tensor: torch.Tensor,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Convert a [3, H, W] tensor to a [H, W, 3] uint8 CPU tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape [3, H, W], got {tuple(tensor.shape)}")

    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 channels in dimension 0, got {tensor.shape[0]}")

    min_val, max_val = value_range
    if max_val <= min_val:
        raise ValueError(f"Invalid value_range: {value_range}")

    tensor = tensor.detach().cpu().float()
    tensor = (tensor - min_val) / (max_val - min_val)
    tensor = tensor.clamp(0.0, 1.0)

    tensor = tensor.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    tensor = (tensor * 255.0).round().to(torch.uint8)

    return tensor


def save_image(
    image: torch.Tensor,
    path: PathLike,
    *,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> None:
    """
    Save a color image tensor as a PNG/JPEG/etc.

    Args:
        image: Tensor shaped [3, H, W].
        path: Output image path, for example "image.png".
        value_range: Numeric range of input tensor values.
            Use (0, 1) for normalized tensors.
            Use (-1, 1) for tanh-style tensors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = _to_uint8_image(image, value_range=value_range).numpy()
    Image.fromarray(image_uint8, mode="RGB").save(path)


def save_video(
    video: torch.Tensor,
    path: PathLike,
    *,
    fps: int = 30,
    value_range: Tuple[float, float] = (0.0, 1.0),
    codec: str = "libx264",
    quality: int = 8,
) -> None:
    """
    Save a color video tensor as MP4/MOV/etc.

    Args:
        video: Tensor shaped [T, 3, H, W].
        path: Output video path, for example "video.mp4" or "video.mov".
        fps: Frames per second.
        value_range: Numeric range of input tensor values.
            Use (0, 1) for normalized tensors.
            Use (-1, 1) for tanh-style tensors.
        codec: Video codec. "libx264" is a good default for MP4/MOV.
        quality: imageio/ffmpeg quality setting. Usually 5-10.
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(video)!r}")

    if video.ndim != 4:
        raise ValueError(f"Expected video shape [T, 3, H, W], got {tuple(video.shape)}")

    if video.shape[1] != 3:
        raise ValueError(f"Expected 3 channels in dimension 1, got {video.shape[1]}")

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        _to_uint8_image(frame, value_range=value_range).numpy()
        for frame in video
    ]

    iio.imwrite(
        path,
        frames,
        fps=fps,
        codec=codec,
        quality=quality,
    )
