"""Safe single, multiple, and folder-based fingerprint image loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class LoadedImage:
    filename: str
    image: np.ndarray
    source: str | None = None


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported image format: {Path(filename).suffix or 'no extension'}")


def _decode_image(data: bytes, filename: str) -> np.ndarray:
    _validate_extension(filename)
    if not data:
        raise ValueError("image file is empty")
    encoded = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("image is invalid or corrupted")
    if decoded.ndim == 2:
        return decoded.astype(np.uint8, copy=False)
    if decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def load_image_bytes(data: bytes, filename: str) -> LoadedImage:
    """Decode uploaded bytes to the project's RGB/grayscale NumPy representation."""
    return LoadedImage(filename=Path(filename).name, image=_decode_image(data, filename))


def load_image(path: str | Path) -> LoadedImage:
    """Load one image without relying on OpenCV's Unicode-path handling."""
    image_path = Path(path)
    _validate_extension(image_path.name)
    data = image_path.read_bytes()
    return LoadedImage(
        filename=image_path.name,
        image=_decode_image(data, image_path.name),
        source=str(image_path),
    )


def load_multiple_images(
    sources: Iterable[str | Path | tuple[str, bytes]],
) -> tuple[list[LoadedImage], list[dict[str, str]]]:
    """Load independent sources; one invalid image does not stop the remaining batch."""
    loaded: list[LoadedImage] = []
    errors: list[dict[str, str]] = []
    for source in sources:
        name = source[0] if isinstance(source, tuple) else str(source)
        try:
            item = load_image_bytes(source[1], source[0]) if isinstance(source, tuple) else load_image(source)
            loaded.append(item)
        except (OSError, ValueError) as exc:
            errors.append({"filename": Path(name).name, "error": str(exc)})
    return loaded, errors


def load_images_from_folder(
    folder: str | Path,
    recursive: bool = False,
) -> tuple[list[LoadedImage], list[dict[str, str]]]:
    """Load all supported images in a local folder, ignoring unsupported files."""
    directory = Path(folder).expanduser()
    if not directory.exists() or not directory.is_dir():
        raise ValueError("folder does not exist or is not a directory")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    paths = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )
    return load_multiple_images(paths)

