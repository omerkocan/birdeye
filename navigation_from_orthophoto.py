import os
import csv
import math
from typing import Tuple, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# =============================
# CONFIGURATION
# =============================
CONFIG = dict(
    map_image_path = "map.jpg",
    gsd_m_per_px   = 0.294985,  # 100 m / 339 px

    # --- CAMERA & FLIGHT ---
    altitude_m = 500.0,   # Above Ground Level (m)
    fov_deg    = 45.0,    # diagonal Field of View (deg)
    yaw_deg    = 0.0,
    roll_deg   = 0.0,
    pitch_deg  = 0.0,

    frame_rate_hz = 2.0,
    speed_mps     = 30.0,
    duration_s    = 8.0,
    start_xy_m    = (180.0, 180.0),

    # --- LOCALIZATION ---
    use_orb    = True,
    n_features = 1400,
    ratio_test = 0.75,
    ransac_reproj_thresh_px = 3.0,

    # --- OUTPUTS ---
    out_dir     = "outputs",
    save_frames = True,

    # --- GEO REFERENCE (top-left pixel) ---
    lat0_deg = 38.010886,  # 38°00'39.19" K
    lon0_deg = 32.518552,  # 32°31'06.79" D

    # --- POINT AND LABEL SIZE ---
    annot_font_size_px = 50,
    annot_margin_px    = 12,
    annot_dot_radius   = 7,
)

# =============================
# HELPERS
# =============================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def meters_to_pixels(xy_m: Tuple[float,float], gsd: float) -> Tuple[float,float]:
    return (xy_m[0] / gsd, xy_m[1] / gsd)

def pixels_to_meters(xy_px: Tuple[float,float], gsd: float) -> Tuple[float,float]:
    return (xy_px[0] * gsd, xy_px[1] * gsd)

def footprint_pixels(altitude_m: float, fov_deg: float, gsd_m_per_px: float, map_img_shape) -> Tuple[int,int]:
    diag_m = 2.0 * altitude_m * math.tan(math.radians(fov_deg) * 0.5)
    side_m = diag_m / math.sqrt(2.0)
    side_px = max(32, int(round(side_m / gsd_m_per_px)))
    H, W = map_img_shape[:2]
    side_px = min(side_px, min(H, W)-4)
    return side_px, side_px

def crop_from_center(img: np.ndarray, center_xy_px: Tuple[float,float], crop_wh: Tuple[int,int]):
    cx, cy = int(round(center_xy_px[0])), int(round(center_xy_px[1]))
    w, h = crop_wh
    x0 = max(0, cx - w//2)
    y0 = max(0, cy - h//2)
    x0 = max(0, min(x0, img.shape[1]-w))
    y0 = max(0, min(y0, img.shape[0]-h))
    x1 = x0 + w
    y1 = y0 + h
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

def match_template(map_img: np.ndarray, frame: np.ndarray):
    res = cv2.matchTemplate(map_img, frame, cv2.TM_CCORR_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_loc, max_val

def orb_localize(map_img: np.ndarray, frame: np.ndarray, n_features=1200, ratio_test=0.75, ransac_thresh=3.0):
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7, scaleFactor=1.2, edgeThreshold=15)
    kpp, desp = orb.detectAndCompute(frame, None)
    kpm, desm = orb.detectAndCompute(map_img, None)
    if desp is None or desm is None or len(kpp) < 8 or len(kpm) < 8:
        return None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desp, desm, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, 0.0
    src = np.float32([kpp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpm[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None, 0.0
    h, w = frame.shape[:2]
    center = np.array([[[w/2.0, h/2.0]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(center, H)
    cx, cy = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    inliers = int(mask.sum()) if mask is not None else 0
    conf = min(1.0, inliers/30.0)
    top_left = (int(round(cx - w/2.0)), int(round(cy - h/2.0)))
    return top_left, conf


def pix_to_latlon(x_px: float, y_px: float, gsd: float, lat0_deg: Optional[float], lon0_deg: Optional[float]):
    if lat0_deg is None or lon0_deg is None:
        return None, None
    meters_e = x_px * gsd
    meters_s = y_px * gsd
    dlat = -meters_s / 111_320.0
    dlon = meters_e / (111_320.0 * math.cos(math.radians(lat0_deg)))
    return lat0_deg + dlat, lon0_deg + dlon

def deg_to_dms_str(deg: Optional[float], is_lat: bool = True, sec_precision: int = 2) -> Optional[str]:
    if deg is None:
        return None
    hemi = ('K' if deg >= 0 else 'G') if is_lat else ('D' if deg >= 0 else 'B')
    v = abs(float(deg))
    d = int(v)
    m_float = (v - d) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    s = round(s, sec_precision)
    if s >= 60.0:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        d += 1
    if sec_precision == 0:
        s_fmt = f"{int(s):02d}"
    elif sec_precision == 1:
        s_fmt = f"{s:04.1f}"
    else:
        s_fmt = f"{s:05.2f}"
    return f"{d:02d}°{m:02d}'{s_fmt}\" {hemi}"

# =============================
# MAIN
# =============================

def main():
    cfg = CONFIG

    # prepare outputs
    ensure_dir(cfg["out_dir"])
    frames_dir = os.path.join(cfg["out_dir"], "frames")
    annot_dir  = os.path.join(cfg["out_dir"], "frames_annotated")
    if cfg["save_frames"]:
        ensure_dir(frames_dir)
        ensure_dir(annot_dir)

    # load map
    map_img = cv2.imread(cfg["map_image_path"])
    if map_img is None:
        raise RuntimeError("Map image could not be read at: " + str(cfg["map_image_path"]))
    H, W = map_img.shape[:2]
    print(f"[INFO] Loaded map: {cfg['map_image_path']} shape={map_img.shape} gsd={cfg['gsd_m_per_px']:.6f} m/px")

    # crop size from AGL & FOV
    crop_w, crop_h = footprint_pixels(cfg["altitude_m"], cfg["fov_deg"], cfg["gsd_m_per_px"], map_img.shape)
    print(f"[INFO] Crop size: {crop_w} x {crop_h} px")

    # build straight path (eastward) in meters
    n_frames = int(cfg["duration_s"] * cfg["frame_rate_hz"])
    dt = 1.0 / cfg["frame_rate_hz"]
    path_xy_m: List[Tuple[float,float]] = []
    for k in range(n_frames):
        t = k * dt
        x = cfg["start_xy_m"][0] + cfg["speed_mps"] * t
        y = cfg["start_xy_m"][1] + 0.0
        path_xy_m.append((x, y))

    # render frames (GT centers in px, with clamping to keep full crop inside)
    gt_centers_px: List[Tuple[float,float]] = []
    frames: List[np.ndarray] = []
    crop_boxes: List[Tuple[int,int,int,int]] = []
    for i, (xm, ym) in enumerate(path_xy_m):
        cx_px, cy_px = meters_to_pixels((xm, ym), cfg["gsd_m_per_px"])
        cx_px = np.clip(cx_px, crop_w//2, W - crop_w//2 - 1)
        cy_px = np.clip(cy_px, crop_h//2, H - crop_h//2 - 1)
        frame, box = crop_from_center(map_img, (cx_px, cy_px), (crop_w, crop_h))
        frames.append(frame)
        crop_boxes.append(box)
        gt_centers_px.append((cx_px, cy_px))
        if cfg["save_frames"]:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.png"), frame)

    # localize each frame
    est_top_lefts: List[Tuple[int,int]] = []
    method_used: List[str] = []
    scores: List[float] = []
    for i, frame in enumerate(frames):
        tl_tm, score_tm = match_template(map_img, frame)
        best_tl = tl_tm
        best_score = float(score_tm)
        used = "TM"
        if cfg["use_orb"]:
            tl_orb, conf = orb_localize(map_img, frame, cfg["n_features"], cfg["ratio_test"], cfg["ransac_reproj_thresh_px"])
            if tl_orb is not None and conf > 0.2:
                if conf + 0.15 > best_score:
                    best_tl = tl_orb
                    best_score = conf
                    used = "ORB"
        est_top_lefts.append(best_tl)
        method_used.append(used)
        scores.append(best_score)

    # centers from top-left
    est_centers_px: List[Tuple[float,float]] = []
    w, h = frames[0].shape[1], frames[0].shape[0]
    for (etl_x, etl_y) in est_top_lefts:
        est_cx = etl_x + w/2.0
        est_cy = etl_y + h/2.0
        est_centers_px.append((est_cx, est_cy))

    # errors in meters
    errors_m: List[float] = []
    for gt, est in zip(gt_centers_px, est_centers_px):
        dx_px = est[0] - gt[0]
        dy_px = est[1] - gt[1]
        dx_m, dy_m = pixels_to_meters((dx_px, dy_px), cfg["gsd_m_per_px"])
        errors_m.append(math.hypot(dx_m, dy_m))

    # ============ FRAME ANNOTATION ============
    if cfg["save_frames"]:
        def _load_font(px=24):
            try:
                ttf_path = fm.findfont("DejaVu Sans", fallback_to_default=True)
                if os.path.exists(ttf_path):
                    return ImageFont.truetype(ttf_path, px)
            except Exception:
                pass
            for name in ("DejaVuSans.ttf", "Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                try:
                    return ImageFont.truetype(name, px)
                except Exception:
                    continue
            return ImageFont.load_default()

        font_px   = int(cfg.get("annot_font_size_px", 24))
        margin_px = int(cfg.get("annot_margin_px", 10))
        dot_r     = int(cfg.get("annot_dot_radius", 6))
        annot_font = _load_font(font_px)

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img_pil)

            fh, fw = frame.shape[:2]
            x0, y0, x1, y1 = crop_boxes[i]

            est_cx_map, est_cy_map = est_centers_px[i]
            px = int(round(est_cx_map - x0))
            py = int(round(est_cy_map - y0))
            px = max(0, min(fw - 1, px))
            py = max(0, min(fh - 1, py))

            draw.ellipse((px - dot_r, py - dot_r, px + dot_r, py + dot_r), fill=(255, 0, 0))

            if cfg["lat0_deg"] is not None and cfg["lon0_deg"] is not None:
                lat, lon = pix_to_latlon(est_cx_map, est_cy_map, cfg["gsd_m_per_px"], cfg["lat0_deg"], cfg["lon0_deg"])
                lat_dms = deg_to_dms_str(lat,  is_lat=True,  sec_precision=2)
                lon_dms = deg_to_dms_str(lon,  is_lat=False, sec_precision=2)
                label = f"{lat_dms}, {lon_dms}"

                try:
                    bbox = draw.textbbox((0, 0), label, font=annot_font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except Exception:
                    tw = draw.textlength(label, font=annot_font)
                    th = annot_font.size

                tx = int(px - tw / 2)
                ty = int(py - dot_r - margin_px - th)
                tx = max(2, min(tx, fw - int(tw) - 2))
                ty = max(2, ty)

                draw.text((tx + 1, ty + 1), label, font=annot_font, fill=(0, 0, 0))
                draw.text((tx, ty),           label, font=annot_font, fill=(255, 0, 0))

            annotated_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(annot_dir, f"frame_{i:04d}.png"), annotated_bgr)
    # =================================================================

    # CSV export
    csv_path = os.path.join(cfg["out_dir"], "estimates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        header = [
            "frame","gt_x_px","gt_y_px","est_x_px","est_y_px","err_m","method","score"
        ]
        if cfg["lat0_deg"] is not None and cfg["lon0_deg"] is not None:
            header += ["lat_est","lon_est","lat_est_dms","lon_est_dms"]
        wcsv.writerow(header)
        for i, (gt, est, e, m, s) in enumerate(zip(gt_centers_px, est_centers_px, errors_m, method_used, scores)):
            row = [i, gt[0], gt[1], est[0], est[1], e, m, s]
            if cfg["lat0_deg"] is not None and cfg["lon0_deg"] is not None:
                lat, lon = pix_to_latlon(est[0], est[1], cfg["gsd_m_per_px"], cfg["lat0_deg"], cfg["lon0_deg"])
                lat_dms = deg_to_dms_str(lat,  is_lat=True,  sec_precision=2)
                lon_dms = deg_to_dms_str(lon,  is_lat=False, sec_precision=2)
                row += [lat, lon, lat_dms, lon_dms]
            wcsv.writerow(row)
    print(f"[INFO] Wrote {csv_path}")

    fig = plt.figure(figsize=(8, 6))
    plt.clf()

    gsd = float(CONFIG["gsd_m_per_px"])
    H, W = map_img.shape[:2]

    extent_m = [0, W * gsd, H * gsd, 0]  # [x_min, x_max, y_min, y_max]
    plt.imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB),
               extent=extent_m, origin='upper')

    gtx_m = [float(p[0]) * gsd for p in gt_centers_px]
    gty_m = [float(p[1]) * gsd for p in gt_centers_px]
    ex_m = [float(p[0]) * gsd for p in est_centers_px]
    ey_m = [float(p[1]) * gsd for p in est_centers_px]

    plt.plot(gtx_m, gty_m, '-', linewidth=2, label='GT path')
    plt.plot(ex_m, ey_m, '.', label='Estimated')

    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.title("Localization on orthoimage (GT vs Estimated)")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    out_plot = os.path.join(CONFIG["out_dir"], "trajectory_plot.png")
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    print(f"[INFO] Wrote {out_plot}")

    mean_err = float(np.mean(errors_m)) if errors_m else float('nan')
    p95_err  = float(np.percentile(errors_m, 95)) if errors_m else float('nan')
    print(f"[STATS] mean err = {mean_err:.3f} m, p95 err = {p95_err:.3f} m, frames={len(errors_m)}")
    print(f"[NOTE] Crop size (px): {w} x {h} from altitude={cfg['altitude_m']} m, fov={cfg['fov_deg']} deg")

if __name__ == "__main__":
    main()
