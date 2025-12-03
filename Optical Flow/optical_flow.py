import cv2, numpy as np, argparse, time, csv
from scipy.spatial.distance import cdist
from collections import namedtuple
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Using simple connected-components clustering.")

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--interactive_roi", action="store_true", help="draw ROI on first frame")
    p.add_argument("--cells", type=int, nargs=2, default=[10,8], help="grid cells (cols rows)")
    p.add_argument("--kpc", type=int, default=20, help="max corners per cell")
    p.add_argument("--ql", type=float, default=0.01, help="qualityLevel for GFTT (increased for stability)")
    p.add_argument("--minDist", type=int, default=5, help="minDist for GFTT (increased for better spacing)")
    p.add_argument("--fb_thresh", type=float, default=3.0, help="forward-backward error px (relaxed for deformation)")
    p.add_argument("--k_nn", type=int, default=4, help="neighbors for edge-strain")
    p.add_argument("--csv", type=str, default="vector_log.csv")
    p.add_argument("--a", type=float, default=1.0, help="force scale (after calibration)")
    p.add_argument("--b", type=float, default=0.0, help="force offset")
    p.add_argument("--min_features", type=int, default=30, help="minimum features before gradual reseed")
    p.add_argument("--reseed_ratio", type=float, default=0.3, help="fraction of features to reseed when low")
    p.add_argument("--of_winSize", type=int, default=31, help="optical flow window size (larger for big deformations)")
    p.add_argument("--of_maxLevel", type=int, default=4, help="optical flow pyramid levels")
    p.add_argument("--deform_k_mad", type=float, default=2.0, help="MAD multiplier for deformation threshold (lower=more sensitive)")
    p.add_argument("--deform_min_cluster", type=int, default=3, help="minimum points per deformation cluster")
    p.add_argument("--deform_eps", type=float, default=20.0, help="DBSCAN eps in pixels for clustering")
    return p.parse_args()
def robust_normalize(arr, eps=1e-6):
    """Normalize by robust max (median + 3*MAD)."""
    if arr.size == 0: return arr
    m = np.median(arr)
    mad = np.median(np.abs(arr - m)) + eps
    vmax = max(eps, m + 3*1.4826*mad)
    v = np.clip(arr / vmax, 0.0, 1.0)
    return v

def make_disp_heatmap(HW, pts_xy, values, roi, sigma_px=15):
    """
    HW: (H, W)
    pts_xy: (N,2) float pixel coords (x,y)
    values: (N,) nonneg scalars (e.g., |U|)
    roi: (x,y,w,h) – we only draw inside ROI
    sigma_px: blur (controls smoothness)
    Returns BGR heatmap image.
    """
    H, W = HW
    x, y, w, h = roi
    canvas = np.zeros((H, W), dtype=np.float32)

    # normalize values robustly → 0..1
    v = robust_normalize(values.astype(np.float32))

    # splat: draw small disks at each point with intensity = value
    r = max(1, sigma_px // 4)  # small disk radius
    for (px, py), val in zip(pts_xy, v):
        ix = int(round(px)); iy = int(round(py))
        if ix < x or ix >= x+w or iy < y or iy >= y+h: 
            continue
        cv2.circle(canvas, (ix, iy), r, float(val), thickness=-1)

    # smooth to make a field
    k = int(max(3, sigma_px*2+1))
    heat = cv2.GaussianBlur(canvas, (k|1, k|1), sigmaX=sigma_px, sigmaY=sigma_px)

    # to color
    heat_u8 = np.clip(heat*255.0, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(heat_u8, cmap)
    return heat_bgr

def clahe(gray):
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return cla.apply(gray)

def pick_grid_features(gray, roi, cells_xy=(10,8), kpc=20, ql=0.01, minDist=5, useHarris=True, exclude_pts=None, exclude_radius=10):
    """
    Pick features from grid cells, optionally excluding areas around existing points.
    exclude_pts: (N,2) array of points to avoid
    exclude_radius: minimum distance from excluded points
    """
    x,y,w,h = roi
    patch = gray[y:y+h, x:x+w]
    H,W = patch.shape
    cx,cy = cells_xy
    sx,sy = max(1,W//cx), max(1,H//cy)
    pts=[]
    
    # Create exclusion mask if needed
    exclude_mask = None
    if exclude_pts is not None and len(exclude_pts) > 0:
        exclude_mask = np.ones((H, W), dtype=np.uint8) * 255
        for pt in exclude_pts:
            px, py = int(pt[0] - x), int(pt[1] - y)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(exclude_mask, (px, py), exclude_radius, 0, -1)
    
    for j in range(cy):
        for i in range(cx):
            rx, ry = i*sx, j*sy
            sub = patch[ry:ry+sy, rx:rx+sx]
            if sub.size < 25: continue
            
            # Apply exclusion mask to subregion if available
            if exclude_mask is not None:
                sub_mask = exclude_mask[ry:ry+sy, rx:rx+sx]
                if np.sum(sub_mask) < sub_mask.size * 0.3:  # Too much exclusion
                    continue
            else:
                sub_mask = None
            
            c = cv2.goodFeaturesToTrack(sub, maxCorners=kpc, qualityLevel=ql, minDistance=minDist,
                                        blockSize=7, useHarrisDetector=useHarris, k=0.04, mask=sub_mask)
            if c is None: continue
            c = c.reshape(-1,2)
            c[:,0] += rx + x; c[:,1] += ry + y
            pts.append(c)
    if not pts: return None
    return np.vstack(pts).astype(np.float32)

def fb_track(prev, curr, p0, fb_thresh, winSize=31, maxLevel=4):
    """
    Forward-backward optical flow tracking with improved parameters for large deformations.
    """
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None,
                                          winSize=(winSize,winSize), maxLevel=maxLevel,
                                          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    if p1 is None or len(p1) == 0: 
        return np.empty((0,1,2),np.float32), np.empty((0,1,2),np.float32), np.array([])
    p0r, st2, err2 = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None,
                                          winSize=(winSize,winSize), maxLevel=maxLevel,
                                          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    ok = (st1.reshape(-1)==1) & (st2.reshape(-1)==1)
    if not np.any(ok):
        return np.empty((0,1,2),np.float32), np.empty((0,1,2),np.float32), np.array([])
    
    fb = np.linalg.norm(p0[ok]-p0r[ok], axis=2).reshape(-1)
    keep = np.zeros_like(ok)
    if fb.size>0:
        # Adaptive threshold: allow larger errors for larger displacements
        max_disp = np.max(np.linalg.norm(p1[ok] - p0[ok], axis=2)) if np.any(ok) else 0
        adaptive_thresh = fb_thresh * (1 + 0.1 * max_disp)  # Scale with displacement
        keep[np.where(ok)[0][fb < adaptive_thresh]] = True
    
    # Return confidence scores (inverse of forward-backward error)
    confidence = np.ones(len(p0)) * 0.5  # Default low confidence
    if fb.size > 0:
        valid_idx = np.where(ok)[0]
        for i, idx in enumerate(valid_idx):
            if keep[idx]:
                confidence[idx] = 1.0 / (1.0 + fb[i])  # Higher confidence for lower error
    
    return p1[keep], p0[keep], confidence[keep]

def remove_global_affine(p0, p1):
    if len(p0)<6: 
        return p0, p1
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=1.8, maxIters=3000, confidence=0.99)
    if M is None: return p0, p1
    p0h = cv2.convertPointsToHomogeneous(p0).reshape(-1,3).astype(np.float32)
    p0_warp = (p0h @ M.T)[:,:2]
    resid = p1 - p0_warp
    p1_corr = p0 + resid
    inl = inl.reshape(-1).astype(bool)
    return p0[inl], p1_corr[inl]

def mean_abs_edge_strain(Xref, Xnow, k=4):
    """Edge strain ε_ij = (|L1|-|L0|)/|L0| averaged over k-NN edges."""
    if len(Xref)<k+1: return np.nan
    d2 = np.sum((Xref[None,:,:]-Xref[:,None,:])**2, axis=2)
    idx = np.argsort(d2, axis=1)[:,1:k+1]
    eps=[]
    for i in range(len(Xref)):
        n = idx[i]
        L0 = Xref[n]-Xref[i]; L1 = Xnow[n]-Xnow[i]
        L0n = np.linalg.norm(L0,axis=1)+1e-6
        L1n = np.linalg.norm(L1,axis=1)
        e = (L1n-L0n)/L0n
        eps.append(np.abs(e))
    eps = np.concatenate(eps)
    # robust mean
    m = np.median(eps); mad = np.median(np.abs(eps-m))+1e-6
    keep = np.abs(eps-m) < 3*1.4826*mad
    return float(np.mean(eps[keep]))

# Region data structure for deformation patches
Region = namedtuple('Region', ['bbox', 'indices', 'mean_disp', 'max_disp', 'n_points'])

def detect_deformation_regions(Xnow, Xref_original, u_mag, roi, params=None):
    """
    Heuristic detection of deformation regions from feature displacements.
    
    This function can be replaced by an ML-based detector in the future.
    
    Args:
        Xnow: (N, 2) current feature positions
        Xref_original: (N, 2) baseline feature positions (must match Xnow indices)
        u_mag: (N,) displacement magnitudes
        roi: (x, y, w, h) ROI bounds
        params: dict with optional parameters:
            - k_mad: multiplier for MAD threshold (default 2.5)
            - min_cluster_size: minimum points per cluster (default 3)
            - eps_px: DBSCAN eps in pixels (default 20)
    
    Returns:
        list[Region]: detected deformation regions
    """
    if params is None:
        params = {}
    k_mad = params.get('k_mad', 2.5)
    min_cluster_size = params.get('min_cluster_size', 3)
    eps_px = params.get('eps_px', 20.0)
    
    if len(Xnow) == 0 or len(u_mag) == 0:
        return []
    
    # Step 1: Robust thresholding
    m = np.median(u_mag)
    mad = np.median(np.abs(u_mag - m)) + 1e-6
    threshold = m + k_mad * 1.4826 * mad  # MAD scaling factor
    
    strong_mask = u_mag > threshold
    if not np.any(strong_mask):
        return []
    
    X_strong = Xnow[strong_mask]
    u_strong = u_mag[strong_mask]
    strong_indices = np.where(strong_mask)[0]
    
    # Step 2: Clustering
    # Compute image bounds for bounding box clipping
    x, y, w, h = roi
    img_h = max(int(np.max(X_strong[:, 1])) + 10, y + h) if len(X_strong) > 0 else y + h
    img_w = max(int(np.max(X_strong[:, 0])) + 10, x + w) if len(X_strong) > 0 else x + w
    
    if HAS_SKLEARN and len(X_strong) >= min_cluster_size:
        # Use DBSCAN
        clustering = DBSCAN(eps=eps_px, min_samples=min_cluster_size).fit(X_strong)
        labels = clustering.labels_
    else:
        # Fallback: simple connected components in image space
        # Create a binary image and use cv2.connectedComponents
        # Use full eps_px as radius to ensure connectivity
        binary = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for pt in X_strong:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < img_w and 0 <= py < img_h:
                cv2.circle(binary, (px, py), int(eps_px), 255, -1)  # Use full eps_px as radius
        
        num_labels, labels_img = cv2.connectedComponents(binary)
        labels = np.zeros(len(X_strong), dtype=int) - 1
        
        for i, pt in enumerate(X_strong):
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < img_w and 0 <= py < img_h:
                labels[i] = labels_img[py, px] - 1  # -1 because 0 is background
    
    # Step 3: Extract regions
    regions = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = X_strong[cluster_mask]
        cluster_disp = u_strong[cluster_mask]
        cluster_indices = strong_indices[cluster_mask]
        
        if len(cluster_points) < min_cluster_size:
            continue
        
        # Compute bounding box
        x_min = int(np.min(cluster_points[:, 0]))
        y_min = int(np.min(cluster_points[:, 1]))
        x_max = int(np.max(cluster_points[:, 0]))
        y_max = int(np.max(cluster_points[:, 1]))
        
        # Add small padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_w, x_max + padding)
        y_max = min(img_h, y_max + padding)
        
        bbox = (x_min, y_min, x_max, y_max)
        mean_disp = float(np.mean(cluster_disp))
        max_disp = float(np.max(cluster_disp))
        
        regions.append(Region(
            bbox=bbox,
            indices=cluster_indices,
            mean_disp=mean_disp,
            max_disp=max_disp,
            n_points=len(cluster_points)
        ))
    
    return regions

def find_available_cameras(max_test=5):
    """Test camera indices to find available cameras."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def main():
    a = args()
    cap = cv2.VideoCapture(a.cam)  # Use default backend (cross-platform)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {a.cam}")
        print("\nSearching for available cameras...")
        available = find_available_cameras()
        if available:
            print(f"✓ Found {len(available)} available camera(s): {available}")
            print(f"\nTry running with one of these:")
            for cam_idx in available:
                print(f"  python optical_flow.py --cam {cam_idx}")
        else:
            print("✗ No cameras found. Please check:")
            print("  - Camera is connected")
            print("  - Camera permissions (try: sudo chmod 666 /dev/video*)")
            print("  - No other program is using the camera")
        cap.release()
        raise RuntimeError(f"Camera {a.cam} not opened")

    ret, frame = cap.read(); 
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read frame from camera {a.cam}")
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray0 = clahe(gray0)
    H,W = gray0.shape

    # ROI selection with better handling
    if a.interactive_roi:
        print("\n" + "="*60)
        print("ROI SELECTION INSTRUCTIONS:")
        print("  1. Click and drag to select the region of interest")
        print("  2. Press SPACE or ENTER to confirm")
        print("  3. Press ESC or 'q' to cancel and use default ROI")
        print("="*60 + "\n")
        
        # Create a copy with instructions overlay
        frame_with_text = frame.copy()
        cv2.putText(frame_with_text, "Drag to select ROI, then press SPACE/ENTER", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_with_text, f"Frame size: {W}x{H} pixels", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # selectROI returns (x, y, w, h) when confirmed, or (0,0,0,0) if cancelled
        # The third parameter (False) means don't show crosshair
        # The fourth parameter (False) means don't center the box
        r = cv2.selectROI("Select ROI - Drag rectangle, then press SPACE/ENTER", 
                         frame_with_text, False, False)
        
        cv2.destroyAllWindows()
        
        # Check if valid ROI was selected
        if r[2] > 0 and r[3] > 0:  # width and height > 0
            x, y, w, h = [int(v) for v in r]
            # Validate ROI is within frame bounds
            x = max(0, min(x, W-1))
            y = max(0, min(y, H-1))
            w = max(10, min(w, W - x))  # Minimum 10 pixels width, ensure fits in frame
            h = max(10, min(h, H - y))  # Minimum 10 pixels height, ensure fits in frame
            print(f"✓ ROI selected: x={x}, y={y}, width={w}, height={h}")
        else:
            # User cancelled or selected invalid ROI
            print("⚠ No valid ROI selected. Using default ROI (70% of frame)")
            w = int(W*0.7); h = int(H*0.7); x = (W-w)//2; y = (H-h)//2
    else:
        w = int(W*0.7); h = int(H*0.7); x = (W-w)//2; y = (H-h)//2
        print(f"Using default ROI: x={x}, y={y}, width={w}, height={h}")
    
    roi = (x,y,w,h)
    print(f"Final ROI: x={x}, y={y}, width={w}, height={h} (covers {100*w*h/(W*H):.1f}% of frame)\n")

    # seed reference points
    p0 = pick_grid_features(gray0, roi, tuple(a.cells), a.kpc, a.ql, a.minDist, useHarris=True)
    if p0 is None or len(p0)<50:
        raise RuntimeError("Not enough features. Improve texture/lighting.")

    # reference positions (undeformed baseline) - can be updated with 'b' key
    Xref_original = p0.reshape(-1,2).copy()
    Xref = Xref_original.copy()  # Working reference (may be updated for matching)
    
    # Track which current features correspond to baseline features
    # This mapping is updated when baseline is reset or features are reseeded
    baseline_feature_map = np.arange(len(p0))  # Initially, all features map to themselves

    # CSV for regular frames
    f = open(a.csv,"w",newline="",encoding="utf-8"); wr = csv.writer(f)
    wr.writerow(["t_s","frame","n_features","S_edge_mean","U_med","F_hat","marked","n_clusters","cluster_stats"])
    
    # CSV for marked frames (max deformation)
    csv_marked = a.csv.replace(".csv", "_marked.csv")
    f_marked = open(csv_marked,"w",newline="",encoding="utf-8"); wr_marked = csv.writer(f_marked)
    wr_marked.writerow(["t_s","frame","n_features","n_clusters","cluster_stats"])
    
    t0 = time.time(); k=0
    baseline_set = True  # True after initial seeding

    prev = gray0; Pprev = p0.reshape(-1,1,2).astype(np.float32)
    feature_confidence = np.ones(len(p0))  # Track confidence for each feature
    
    # State for max deformation frame
    marked_regions = None
    marked_frame_data = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe(gray)

        # track with FB (using improved parameters)
        Pnow, Pkeep, conf = fb_track(prev, gray, Pprev, a.fb_thresh, 
                                     winSize=a.of_winSize, maxLevel=a.of_maxLevel)
        
        # Update confidence scores
        if len(Pnow) > 0:
            feature_confidence = conf
        
        # Gradual reseeding: add new features when count is low, but keep existing ones
        n_tracked = len(Pnow)
        if n_tracked < a.min_features:
            # Find areas without features for reseeding
            existing_pts = Pnow.reshape(-1, 2) if len(Pnow) > 0 else np.empty((0, 2))
            
            # Reseed to fill gaps
            n_to_add = max(50 - n_tracked, int(n_tracked * a.reseed_ratio))
            new_features = pick_grid_features(gray, roi, tuple(a.cells), 
                                             kpc=n_to_add // (a.cells[0] * a.cells[1]) + 1,
                                             ql=a.ql, minDist=a.minDist, 
                                             exclude_pts=existing_pts, exclude_radius=a.minDist*2)
            
            if new_features is not None and len(new_features) > 0:
                # Add new features to existing ones
                new_features_reshaped = new_features.reshape(-1, 1, 2).astype(np.float32)
                if len(Pnow) > 0:
                    Pnow = np.vstack([Pnow, new_features_reshaped])
                    Pkeep = np.vstack([Pkeep, new_features.reshape(-1, 1, 2).astype(np.float32)])
                    # New features start with medium confidence
                    feature_confidence = np.concatenate([feature_confidence, 
                                                       np.ones(len(new_features)) * 0.7])
                else:
                    # Complete reseed only if we have NO features
                    Pnow = new_features_reshaped
                    Pkeep = new_features.reshape(-1, 1, 2).astype(np.float32)
                    feature_confidence = np.ones(len(new_features)) * 0.7
                    # Update working reference but keep original
                    Xref = new_features.copy()
        
        if len(Pnow) == 0:
            prev = gray
            continue

        # remove global affine jiggle
        X0 = Pkeep.reshape(-1,2); X1 = Pnow.reshape(-1,2)
        X0_in, X1_corr = remove_global_affine(X0, X1)
        
        # Match current features to baseline (Xref_original)
        # Compute displacement U = Xnow - Xref_original for matched features
        if len(X1_corr) > 0 and baseline_set and len(Xref_original) > 0:
            # Find nearest neighbors in baseline for each current feature
            dists = cdist(X1_corr, Xref_original)
            nearest_idx = np.argmin(dists, axis=1)
            nearest_dists = dists[np.arange(len(X1_corr)), nearest_idx]
            
            # Match threshold: features within threshold are considered matched to baseline
            # Use adaptive threshold based on expected deformation scale
            # For large deformations, we need a larger matching radius
            match_threshold = max(30.0, np.percentile(nearest_dists, 75) * 1.5)  # Adaptive: 1.5x 75th percentile
            matched_mask = nearest_dists < match_threshold
            
            # For matched features: U = Xnow - Xref_original
            # For unmatched (new) features: U = 0 (no baseline displacement)
            U = np.zeros_like(X1_corr)
            if np.any(matched_mask):
                U[matched_mask] = X1_corr[matched_mask] - Xref_original[nearest_idx[matched_mask]]
            
            # Store which baseline features are matched (for later use)
            baseline_matched_indices = nearest_idx[matched_mask] if np.any(matched_mask) else np.array([], dtype=int)
        else:
            # No baseline set yet, or no features
            U = np.zeros_like(X1_corr)
            matched_mask = np.zeros(len(X1_corr), dtype=bool)
            baseline_matched_indices = np.array([], dtype=int)
        # scalar deformation proxies (using matched features for strain)
        X0_ref_matched = Xref_original[baseline_matched_indices] if len(baseline_matched_indices) > 0 else X1_corr[matched_mask] if np.any(matched_mask) else X1_corr
        X1_matched = X1_corr[matched_mask] if np.any(matched_mask) else X1_corr
        S = mean_abs_edge_strain(X0_ref_matched, X1_matched, k=a.k_nn) if len(X0_ref_matched) > a.k_nn else np.nan
        u_mag = np.linalg.norm(U, axis=1)  # Displacement magnitude
        U_med = float(np.median(u_mag)) if len(u_mag)>0 else 0.0
        F_hat = a.a*S + a.b if not np.isnan(S) else 0.0

        # Handle keyboard input (read at start of processing)
        key = cv2.waitKey(1) & 0xFF
        
        # 'b' key: Set/update baseline
        if key == ord('b'):
            if len(X1_corr) > 0:
                Xref_original = X1_corr.copy()
                baseline_set = True
                marked_regions = None  # Clear marked regions when baseline changes
                print(f"[Frame {k}] Baseline updated with {len(X1_corr)} features")
        
        # 'm' key: Mark max deformation frame
        if key == ord('m'):
            if len(X1_corr) > 0 and baseline_set:
                # Debug: Print displacement statistics
                n_matched = np.sum(matched_mask) if 'matched_mask' in locals() else 0
                u_mag_matched = u_mag[matched_mask] if np.any(matched_mask) else np.array([])
                print(f"[Frame {k}] Displacement stats: n_features={len(X1_corr)}, n_matched={n_matched}")
                if len(u_mag_matched) > 0:
                    print(f"  u_mag: min={np.min(u_mag_matched):.2f}, median={np.median(u_mag_matched):.2f}, max={np.max(u_mag_matched):.2f}, mean={np.mean(u_mag_matched):.2f}")
                else:
                    print(f"  WARNING: No features matched to baseline! Check matching threshold.")
                    print(f"  All u_mag: min={np.min(u_mag):.2f}, median={np.median(u_mag):.2f}, max={np.max(u_mag):.2f}")
                
                # Detect deformation regions
                deform_params = {
                    'k_mad': a.deform_k_mad,
                    'min_cluster_size': a.deform_min_cluster,
                    'eps_px': a.deform_eps
                }
                marked_regions = detect_deformation_regions(X1_corr, Xref_original, u_mag, roi, deform_params)
                
                # Debug: Print threshold info
                if len(u_mag) > 0:
                    m = np.median(u_mag)
                    mad = np.median(np.abs(u_mag - m)) + 1e-6
                    threshold = m + a.deform_k_mad * 1.4826 * mad
                    n_above_thresh = np.sum(u_mag > threshold)
                    print(f"  Threshold (median + {a.deform_k_mad}*MAD): {threshold:.2f}, features above: {n_above_thresh}")
                    
                    # Debug clustering
                    if n_above_thresh > 0:
                        X_strong_debug = X1_corr[u_mag > threshold]
                        if len(X_strong_debug) > 1:
                            # Compute pairwise distances using cdist (already imported)
                            pairwise_dists_matrix = cdist(X_strong_debug, X_strong_debug)
                            # Get upper triangle (avoid duplicates and diagonal)
                            n_pts = len(X_strong_debug)
                            pairwise_dists = pairwise_dists_matrix[np.triu_indices(n_pts, k=1)]
                            print(f"  Clustering: eps={a.deform_eps}px, min_cluster={a.deform_min_cluster}")
                            print(f"  Pairwise distances: min={np.min(pairwise_dists):.1f}, median={np.median(pairwise_dists):.1f}, max={np.max(pairwise_dists):.1f}")
                            print(f"  Features within eps: {np.sum(pairwise_dists <= a.deform_eps)}/{len(pairwise_dists)} pairs")
                
                # Prepare cluster stats string
                cluster_stats = []
                for r in marked_regions:
                    cluster_stats.append(f"d_mean={r.mean_disp:.2f},d_max={r.max_disp:.2f},n={r.n_points}")
                cluster_stats_str = "|".join(cluster_stats) if cluster_stats else ""
                
                # Log to marked frames CSV
                wr_marked.writerow([
                    time.time() - t0,
                    k,
                    len(X1_corr),
                    len(marked_regions),
                    cluster_stats_str
                ])
                f_marked.flush()
                
                print(f"[Frame {k}] Marked max deformation: {len(marked_regions)} regions detected")
                if len(marked_regions) == 0 and n_above_thresh > 0:
                    print(f"  Tip: Features above threshold but not clustering. Try:")
                    print(f"    - Increase --deform_eps (current: {a.deform_eps}) to group distant features")
                    print(f"    - Decrease --deform_min_cluster (current: {a.deform_min_cluster}) to allow smaller clusters")
                    print(f"    - Or lower --deform_k_mad (current: {a.deform_k_mad}) to get more features")
                elif len(marked_regions) == 0:
                    print(f"  Tip: Try lowering --deform_k_mad (current: {a.deform_k_mad}) to detect smaller deformations")
                marked_frame_data = {
                    'regions': marked_regions,
                    'X1_corr': X1_corr.copy(),
                    'u_mag': u_mag.copy(),
                    'frame': frame.copy()
                }
        
        # draw
        disp = frame.copy()
        cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 1)
        
        # Draw features with color coding based on confidence
        for i, (X, d) in enumerate(zip(X1_corr.astype(int), U.astype(int))):
            tip = (int(X[0]+d[0]), int(X[1]+d[1]))
            # Color: green (high conf) -> yellow (medium) -> red (low)
            conf_val = feature_confidence[i] if i < len(feature_confidence) else 0.5
            if conf_val > 0.8:
                color = (0, 255, 0)  # Green
            elif conf_val > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            cv2.arrowedLine(disp, tuple(X), tip, color, 1, tipLength=0.3)
            cv2.circle(disp, tuple(X), 2, color, -1)
        
        # Draw bounding boxes for marked regions (or current frame if 'm' was just pressed)
        regions_to_draw = marked_regions if marked_regions is not None else []
        if key == ord('m') and marked_regions is not None:
            regions_to_draw = marked_regions
        
        for r in regions_to_draw:
            x_min, y_min, x_max, y_max = r.bbox
            # Draw bounding box in green
            cv2.rectangle(disp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Draw label with stats
            label = f"d_mean={r.mean_disp:.1f}px, d_max={r.max_disp:.1f}px, n={r.n_points}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y_min - 5, label_size[1] + 5)
            cv2.rectangle(disp, (x_min, label_y - label_size[1] - 5), 
                         (x_min + label_size[0] + 5, label_y + 5), (0, 255, 0), -1)
            cv2.putText(disp, label, (x_min + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Status text
        baseline_status = "BASELINE SET" if baseline_set else "NO BASELINE (press 'b')"
        marked_status = f" | MARKED: {len(marked_regions)} regions" if marked_regions is not None else ""
        txt = f"N:{len(X1_corr)}  S:{S:.4f}  |U|med:{U_med:.3f}  F̂:{F_hat:.2f}  {baseline_status}{marked_status}"
        cv2.putText(disp, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(disp, "Press 'b'=baseline, 'm'=mark max deformation, 'q'=quit", 
                   (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("vector-lines", disp)

        # --- build a separate deformation heatmap window ---
        heat_bgr = make_disp_heatmap(
            (frame.shape[0], frame.shape[1]),
            X1_corr,            # use current positions as anchors
            u_mag,              # magnitude of displacement
            roi,
            sigma_px=15         # adjust smoothness (10–20 works well)
        )
        
        # Overlay bounding boxes on heatmap if regions are marked
        if regions_to_draw:
            for r in regions_to_draw:
                x_min, y_min, x_max, y_max = r.bbox
                cv2.rectangle(heat_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("deform-heat", heat_bgr)

        # log regular frame
        cluster_stats_str = ""
        n_clusters = 0
        if marked_regions is not None:
            n_clusters = len(marked_regions)
            cluster_stats = [f"d_mean={r.mean_disp:.2f},d_max={r.max_disp:.2f},n={r.n_points}" 
                           for r in marked_regions]
            cluster_stats_str = "|".join(cluster_stats)
        
        wr.writerow([time.time()-t0, k, int(len(X1_corr)), S, U_med, F_hat, 
                    "marked" if key == ord('m') else "", n_clusters, cluster_stats_str])
        f.flush()

        prev = gray
        Pprev = Pnow.copy()
        k += 1
        
        if key in (27, ord('q')): break

    f.close()
    f_marked.close()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
