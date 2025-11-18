import cv2, numpy as np, argparse, time, csv
from scipy.spatial.distance import cdist

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

def main():
    a = args()
    cap = cv2.VideoCapture(a.cam)  # Use default backend (cross-platform)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened(): raise RuntimeError("Camera not opened")

    ret, frame = cap.read(); 
    if not ret: raise RuntimeError("No frame")
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray0 = clahe(gray0)
    H,W = gray0.shape

    if a.interactive_roi:
        r = cv2.selectROI("Select ROI (ENTER)", frame, False, True); cv2.destroyAllWindows()
        x,y,w,h = [int(v) for v in r]
        if w==0 or h==0: x,y,w,h = 0,0,W,H
    else:
        w = int(W*0.7); h = int(H*0.7); x = (W-w)//2; y = (H-h)//2
    roi = (x,y,w,h)

    # seed reference points
    p0 = pick_grid_features(gray0, roi, tuple(a.cells), a.kpc, a.ql, a.minDist, useHarris=True)
    if p0 is None or len(p0)<50:
        raise RuntimeError("Not enough features. Improve texture/lighting.")

    # reference positions (undeformed) - NEVER RESET THIS
    Xref_original = p0.reshape(-1,2).copy()
    Xref = Xref_original.copy()  # Working reference (may be updated for matching)

    # CSV
    f = open(a.csv,"w",newline="",encoding="utf-8"); wr = csv.writer(f)
    wr.writerow(["t_s","frame","n_features","S_edge_mean","U_med","F_hat"])
    t0 = time.time(); k=0

    prev = gray0; Pprev = p0.reshape(-1,1,2).astype(np.float32)
    feature_confidence = np.ones(len(p0))  # Track confidence for each feature

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
        
        # Match current features to original reference frame
        # For features that were in original reference, compute displacement from original
        # For new features, use current position as reference
        if len(X1_corr) > 0:
            # Find nearest neighbors in original reference for each current feature
            if len(Xref_original) > 0:
                dists = cdist(X1_corr, Xref_original)
                nearest_idx = np.argmin(dists, axis=1)
                nearest_dists = dists[np.arange(len(X1_corr)), nearest_idx]
                
                # Use original reference for features close to original positions
                # For new features far from original, use current as reference
                use_original = nearest_dists < 20  # 20 pixel threshold
                X0_ref = np.zeros_like(X1_corr)
                X0_ref[use_original] = Xref_original[nearest_idx[use_original]]
                X0_ref[~use_original] = X1_corr[~use_original]  # New features: self-reference
            else:
                X0_ref = X1_corr  # Fallback
        else:
            X0_ref = X1_corr

        # vectors U from reference → current
        U = (X1_corr - X0_ref)    # Displacement vectors
        # scalar deformation proxies
        S = mean_abs_edge_strain(X0_ref, X1_corr, k=a.k_nn) if len(X0_ref) > a.k_nn else np.nan
        U_med = float(np.median(np.linalg.norm(U,axis=1))) if len(U)>0 else 0.0
        F_hat = a.a*S + a.b if not np.isnan(S) else 0.0

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
        
        txt = f"N:{len(X1_corr)}  S:{S:.4f}  |U|med:{U_med:.3f}  F̂:{F_hat:.2f}  Conf:{np.mean(feature_confidence):.2f}"
        cv2.putText(disp, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("vector-lines", disp)
        
        # vectors from reference to current for heatmap
        u_mag = np.linalg.norm(U, axis=1)          # (M,)

        # --- build a separate deformation heatmap window ---
        heat_bgr = make_disp_heatmap(
            (frame.shape[0], frame.shape[1]),
            X1_corr,            # use current positions as anchors
            u_mag,              # magnitude of displacement
            roi,
            sigma_px=15         # adjust smoothness (10–20 works well)
        )

        cv2.imshow("deform-heat", heat_bgr)

        # log
        wr.writerow([time.time()-t0, k, int(len(X1_corr)), S, U_med, F_hat])

        prev = gray
        Pprev = Pnow.copy()
        k += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break

    f.close()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
