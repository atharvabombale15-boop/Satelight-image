import cv2
import numpy as np

def align_t2_to_t1(t1_rgb: np.ndarray, t2_rgb: np.ndarray, max_features: int = 8000, keep_percent: float = 0.2):
    """
    Align t2 image to t1 image using ORB feature matching + homography.
    Inputs: RGB uint8 arrays (H,W,3)
    Returns: aligned_t2 (RGB), H (homography matrix or None), debug dict
    """

    # Convert to grayscale for feature matching
    t1_gray = cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2GRAY)
    t2_gray = cv2.cvtColor(t2_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)

    kpsA, descA = orb.detectAndCompute(t1_gray, None)
    kpsB, descB = orb.detectAndCompute(t2_gray, None)

    if descA is None or descB is None or len(kpsA) < 10 or len(kpsB) < 10:
        return t2_rgb, None, {"reason": "Not enough keypoints"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descA, descB)
    matches = sorted(matches, key=lambda x: x.distance)

    keep = max(10, int(len(matches) * keep_percent))
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype=np.float32)
    ptsB = np.zeros((len(matches), 2), dtype=np.float32)

    for i, m in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # Homography maps B -> A
    H, mask = cv2.findHomography(ptsB, ptsA, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is None:
        return t2_rgb, None, {"reason": "Homography failed"}

    h, w = t1_rgb.shape[:2]
    aligned = cv2.warpPerspective(t2_rgb, H, (w, h), flags=cv2.INTER_LINEAR)

    debug = {
        "num_kp_t1": len(kpsA),
        "num_kp_t2": len(kpsB),
        "num_matches": len(matches),
        "inliers": int(mask.sum()) if mask is not None else None
    }

    return aligned, H, debug