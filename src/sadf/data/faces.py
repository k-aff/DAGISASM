"""Face localisation and cropping.

Default backend is OpenCV's bundled Haar cascade: zero extra downloads,
fully offline. For higher recall install `facenet-pytorch` and set
face.detector: mtcnn. `none` disables detection (used for smoke tests and
datasets that are already tight face crops).
"""
from __future__ import annotations

import numpy as np

CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def _find_cascade_path() -> str | None:
    """Locate the frontal-face cascade XML shipped with the cv2 wheel."""
    import glob
    import os

    import cv2

    # 1) the documented location
    candidate = os.path.join(cv2.data.haarcascades, CASCADE_FILENAME)
    if os.path.exists(candidate):
        return candidate
    # 2) search the installed cv2 package tree (covers odd wheel layouts)
    hits = glob.glob(
        os.path.join(os.path.dirname(cv2.__file__), "**", CASCADE_FILENAME),
        recursive=True,
    )
    return hits[0] if hits else None


def _load_haar_cascade():
    """Load the Haar cascade robustly.

    On some Windows builds `CascadeClassifier(path)` silently produces an
    EMPTY classifier — the file exists and os.path.exists() is True, but
    OpenCV's C++ FileStorage fails to open it (the error shows only the
    bare filename, not the full path). We therefore (1) try the normal
    path load and verify it is non-empty, and (2) fall back to reading the
    XML bytes in Python and handing them to FileStorage via memory, which
    sidesteps the path-parsing layer entirely.
    """
    import cv2

    path = _find_cascade_path()
    if path is None:
        raise RuntimeError(
            f"Could not find {CASCADE_FILENAME} in the OpenCV install. "
            "Reinstall with: pip install 'opencv-python-headless>=4.8,<5'"
        )

    # Attempt 1: normal path load, then verify it actually populated.
    clf = cv2.CascadeClassifier(path)
    if not clf.empty():
        return clf

    # Attempt 2: read bytes ourselves and load from an in-memory FileStorage.
    try:
        with open(path, "rb") as f:
            data = f.read()
        fs = cv2.FileStorage(
            data.decode("utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
        )
        clf = cv2.CascadeClassifier()
        clf.read(fs.getFirstTopLevelNode())
        fs.release()
        if not clf.empty():
            return clf
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Found the cascade at {path} but OpenCV could not load it "
            f"({type(e).__name__}: {e}). A common cause is a non-ASCII "
            "character in the path. Try moving the project to a simple path "
            "like C:\\dev\\project, or set face.detector: mtcnn "
            "(pip install facenet-pytorch)."
        ) from e

    raise RuntimeError(
        f"The cascade at {path} loaded as EMPTY. Move the project to an "
        "ASCII-only path (e.g. C:\\dev\\project), or use face.detector: mtcnn."
    )


class FaceCropper:
    def __init__(
        self,
        detector: str = "haar",
        min_face: int = 40,
        margin: float = 0.30,
    ) -> None:
        self.kind = detector
        self.min_face = int(min_face)
        self.margin = float(margin)
        self._haar = None
        self._mtcnn = None
        if detector == "haar":
            import cv2

            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError(
                    f"Your OpenCV ({cv2.__version__}) has no CascadeClassifier: "
                    "OpenCV 5 moved Haar cascades to the contrib xobjdetect "
                    "module. Fix: pip uninstall -y opencv-python "
                    "opencv-python-headless && "
                    "pip install 'opencv-python-headless>=4.8,<5'"
                )
            self._haar = _load_haar_cascade()
        elif detector == "mtcnn":
            try:
                from facenet_pytorch import MTCNN  # optional dependency
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "face.detector=mtcnn requires `pip install facenet-pytorch`"
                ) from e
            self._mtcnn = MTCNN(keep_all=True, post_process=False)
        elif detector != "none":
            raise ValueError(f"Unknown face detector '{detector}'")

    # ------------------------------------------------------------------
    def _detect(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return the largest face box (x, y, w, h) or None."""
        if self.kind == "none":
            h, w = bgr.shape[:2]
            return (0, 0, w, h)
        if self.kind == "haar":
            import cv2

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.min_face, self.min_face),
            )
            if len(boxes) == 0:
                return None
            return tuple(max(boxes, key=lambda b: b[2] * b[3]))
        # mtcnn
        from PIL import Image
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det, _ = self._mtcnn.detect(Image.fromarray(rgb))
        if det is None or len(det) == 0:
            return None
        x1, y1, x2, y2 = max(det, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        w, h = x2 - x1, y2 - y1
        if min(w, h) < self.min_face:
            return None
        return int(x1), int(y1), int(w), int(h)

    # ------------------------------------------------------------------
    def crop(self, bgr: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Crop the dominant face with margin. Returns (crop, face_found)."""
        crop, found, _ = self.crop_with_box(bgr)
        return crop, found

    def crop_with_box(
        self, bgr: np.ndarray
    ) -> tuple[np.ndarray | None, bool, tuple | None]:
        """Like crop(), but also returns the face box within the CROP.

        The returned box (fx, fy, fw, fh) locates the detected face inside
        the returned crop's coordinate frame, so downstream skin-tone
        estimation can measure only the facial region rather than the whole
        (margin-padded) crop. Box is None when no face was found.
        """
        box = self._detect(bgr)
        if box is None:
            return None, False, None
        x, y, w, h = box
        H, W = bgr.shape[:2]
        mx, my = int(w * self.margin), int(h * self.margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        if x1 <= x0 or y1 <= y0:
            return None, False, None
        crop = bgr[y0:y1, x0:x1]
        # Face box expressed in the CROP's coordinate frame.
        face_box = (x - x0, y - y0, w, h)
        return crop, self.kind != "none", face_box
