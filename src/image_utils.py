"""
image_utils.py
==============
Author: Student 4 (Streamlit UI & Preprocessing Specialist)
Assignment 1: Training Models on MNIST + Interactive Drawing App

Responsibilities:
- Process raw RGBA canvas drawings from streamlit-drawable-canvas.
- Implement the exact MNIST canonical preprocessing pipeline:
    1. Extract stroke intensity into single grayscale channel.
    2. Detect digit bounding box to eliminate arbitrary canvas offsets.
    3. Scale digit to fit inside a 20x20 bounding box (preserving aspect ratio).
    4. Center the 20x20 digit inside a 28x28 image frame with 4-pixel margins
       (using center-of-mass or bounding-box centering).
    5. Normalize pixel intensities to [0.0, 1.0].
- Format tensors for inference:
    - Flattened vector (1, 784) for Classical ML and MLP.
    - Spatial tensor (1, 28, 28, 1) for CNN.
    - 2D array (28, 28) for UI preview visualization.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


def preprocess_canvas_image(
    canvas_rgba: np.ndarray,
    threshold: float = 20.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Transforms raw Streamlit canvas drawing into MNIST-standard 28x28 format.

    Parameters
    ----------
    canvas_rgba : np.ndarray
        Array of shape (H, W, 4) returned by streamlit-drawable-canvas.
    threshold : float
        Pixel intensity threshold to detect drawing strokes above background.

    Returns
    -------
    img_flat : np.ndarray or None
        Normalized shape (1, 784) for Classical/MLP inference. None if canvas is blank.
    img_cnn : np.ndarray or None
        Normalized shape (1, 28, 28, 1) for CNN inference. None if canvas is blank.
    img_2d : np.ndarray or None
        Normalized shape (28, 28) for Streamlit image preview. None if canvas is blank.
    meta : dict
        Metadata including bounding box coordinates and center of mass shift.
    """
    if canvas_rgba is None or canvas_rgba.size == 0:
        return None, None, None, {"status": "empty_input"}

    # 1. Convert RGBA to Grayscale
    # If drawn on black canvas with white brush: RGB channels hold stroke intensity.
    # Alpha channel can also indicate drawn strokes.
    if canvas_rgba.shape[-1] == 4:
        rgb = canvas_rgba[:, :, :3]
        alpha = canvas_rgba[:, :, 3]
        # Use grayscale conversion weighted by alpha
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # In case canvas had transparent background with white brush
        gray = np.where(alpha > 0, gray, 0).astype(np.float32)
    else:
        gray = cv2.cvtColor(canvas_rgba.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. Check for blank canvas
    if np.max(gray) < threshold:
        return None, None, None, {"status": "blank_canvas"}

    # 3. Detect Bounding Box
    coords = cv2.findNonZero((gray > threshold).astype(np.uint8))
    if coords is None:
        return None, None, None, {"status": "no_contour"}

    x, y, w, h = cv2.boundingRect(coords)

    # Filter out tiny specks, dots, or accidental single clicks
    # A real handwritten digit on a 280x280 canvas must have sufficient stroke span and area
    stroke_pixel_count = int(np.count_nonzero(gray > threshold))
    if (w < 20 and h < 20) or stroke_pixel_count < 100:
        return None, None, None, {
            "status": "too_small",
            "message": "Drawing is too small or appears to be just a dot. Please draw a complete digit.",
            "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "pixel_count": stroke_pixel_count,
        }

    digit_crop = gray[y : y + h, x : x + w]

    # 4. Aspect-ratio Preserved Resize to fit in 20x20 box (standard MNIST procedure)
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    digit_resized = cv2.resize(
        digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # 5. Place in 28x28 black canvas (Center of Mass or Geometric Centering)
    canvas_28 = np.zeros((28, 28), dtype=np.float32)

    # Calculate padding to center geometrically
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    canvas_28[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = digit_resized

    # Optional: Fine Center of Mass adjustment (LeCun canonical method)
    canvas_28 = center_by_center_of_mass(canvas_28)

    # 6. Normalize to [0.0, 1.0]
    max_val = np.max(canvas_28)
    if max_val > 0:
        canvas_28 = canvas_28 / 255.0 if max_val > 1.0 else canvas_28
        canvas_28 = np.clip(canvas_28, 0.0, 1.0).astype(np.float32)

    # 7. Format Model Input Tensors
    img_flat = canvas_28.reshape(1, 784)
    img_cnn = canvas_28.reshape(1, 28, 28, 1)
    img_2d = canvas_28

    metadata = {
        "status": "success",
        "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "scaled_shape": (new_w, new_h),
    }

    return img_flat, img_cnn, img_2d, metadata


def center_by_center_of_mass(img: np.ndarray) -> np.ndarray:
    """
    Shifts the 28x28 image such that the center of mass of pixel intensities
    aligns with the geometric center (14, 14), identical to the MNIST dataset construction.
    """
    moments = cv2.moments(img)
    if moments["m00"] == 0:
        return img

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    shift_x = np.round(14.0 - cx).astype(np.float32)
    shift_y = np.round(14.0 - cy).astype(np.float32)

    # Limit shift within +/- 4 pixels to prevent digit clipping
    shift_x = np.clip(shift_x, -4.0, 4.0)
    shift_y = np.clip(shift_y, -4.0, 4.0)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(
        img, M, (28, 28), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return shifted


if __name__ == "__main__":
    print("=== Testing image_utils.py (Student 4) ===")
    # Simulate a 280x280 dummy canvas with a drawn white circle
    dummy_canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    dummy_canvas[:, :, 3] = 255  # Solid alpha
    cv2.circle(dummy_canvas, (140, 140), 50, (255, 255, 255, 255), thickness=15)

    flat, cnn, img2d, meta = preprocess_canvas_image(dummy_canvas)
    print(f"Meta: {meta}")
    print(f"Flattened shape: {flat.shape} | Range: [{flat.min():.2f}, {flat.max():.2f}]")
    print(f"CNN shape:       {cnn.shape}  | Range: [{cnn.min():.2f}, {cnn.max():.2f}]")
    print(f"2D shape:        {img2d.shape}| Range: [{img2d.min():.2f}, {img2d.max():.2f}]")
    print("image_utils.py test passed!")
