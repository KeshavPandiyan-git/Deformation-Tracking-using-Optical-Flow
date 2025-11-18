import cv2, numpy as np, argparse, time, csv

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--interactive_roi", action="store_true", help="draw ROI on first frame")
    p.add_argument("--cells", type=int, nargs=2, default=[10,8], help="grid cells (cols rows)")
    p.add_argument("--kpc", type=int, default=20, help="max corners per cell")
    p.add_argument("--ql", type=float, default=0.004, help="qualityLevel for GFTT")
    p.add_argument("--minDist", type=int, default=3, help="minDist for GFTT")
    p.add_argument("--fb_thresh", type=float, default=1.0, help="forward-backward error px")
    p.add_argument("--k_nn", type=int, default=4, help="neighbors for edge-strain")
    p.add_argument("--csv", type=str, default="vector_log.csv")
    p.add_argument("--a", type=float, default=1.0, help="force scale (after calibration)")
    p.add_argument("--b", type=float, default=0.0, help="force offset")
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

def pick_grid_features(gray, roi, cells_xy=(10,8), kpc=20, ql=0.004, minDist=3, useHarris=True):
    x,y,w,h = roi
    patch = gray[y:y+h, x:x+w]
    H,W = patch.shape
    cx,cy = cells_xy
    sx,sy = max(1,W//cx), max(1,H//cy)
    pts=[]
    for j in range(cy):
        for i in range(cx):
            rx, ry = i*sx, j*sy
            sub = patch[ry:ry+sy, rx:rx+sx]
            if sub.size < 25: continue
            c = cv2.goodFeaturesToTrack(sub, maxCorners=kpc, qualityLevel=ql, minDistance=minDist,
                                        blockSize=7, useHarrisDetector=useHarris, k=0.04)
            if c is None: continue
            c = c.reshape(-1,2)
            c[:,0] += rx + x; c[:,1] += ry + y
            pts.append(c)
    if not pts: return None
    return np.vstack(pts).astype(np.float32)

def fb_track(prev, curr, p0, fb_thresh):
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None,
                                          winSize=(25,25), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    if p1 is None: return np.empty((0,1,2),np.float32), np.empty((0,1,2),np.float32)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None,
                                          winSize=(25,25), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    ok = (st1.reshape(-1)==1) & (st2.reshape(-1)==1)
    fb = np.linalg.norm(p0[ok]-p0r[ok], axis=2).reshape(-1)
    keep = np.zeros_like(ok)
    if fb.size>0:
        keep[np.where(ok)[0][fb<fb_thresh]] = True
    return p1[keep], p0[keep]

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

    # reference positions (undeformed)
    Xref = p0.reshape(-1,2).copy()

    # CSV
    f = open(a.csv,"w",newline="",encoding="utf-8"); wr = csv.writer(f)
    wr.writerow(["t_s","frame","n_features","S_edge_mean","U_med","F_hat"])
    t0 = time.time(); k=0

    prev = gray0; Pprev = p0.reshape(-1,1,2).astype(np.float32)

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe(gray)

        # track with FB
        Pnow, Pkeep = fb_track(prev, gray, Pprev, a.fb_thresh)
        if len(Pnow)<20:
            # hard reseed (also reset reference)
            Pprev = pick_grid_features(gray, roi, tuple(a.cells), a.kpc, a.ql, a.minDist, useHarris=True)
            if Pprev is None or len(Pprev)<50:
                prev = gray; continue
            Xref = Pprev.reshape(-1,2).copy()
            Pprev = Pprev.reshape(-1,1,2).astype(np.float32)
            prev = gray; continue

        # remove global affine jiggle
        X0 = Pkeep.reshape(-1,2); X1 = Pnow.reshape(-1,2)
        X0_in, X1_corr = remove_global_affine(X0, X1)

        # vectors U from reference → current (align sizes via nearest matching)
        # (since we reseed sometimes, use current X0_in as reference for this epoch)
        U = (X1_corr - X0_in)    # vector lines for drawing
        # scalar deformation proxies
        S = mean_abs_edge_strain(X0_in, X1_corr, k=a.k_nn)   # GelSight-like
        U_med = float(np.median(np.linalg.norm(U,axis=1))) if len(U)>0 else 0.0
        F_hat = a.a*S + a.b

        # draw
        disp = frame.copy()
        cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 1)
        for (X, d) in zip(X0_in.astype(int), U.astype(int)):
            tip = (int(X[0]+d[0]), int(X[1]+d[1]))
            cv2.arrowedLine(disp, tuple(X), tip, (0,255,0), 1, tipLength=0.3)
        txt = f"N:{len(X1_corr)}  S:{S:.4f}  |U|med:{U_med:.3f}  F̂:{F_hat:.2f}"
        cv2.putText(disp, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("vector-lines", disp)
        # vectors from reference to current
        U = (X1_corr - X0_in)                      # (M,2)
        u_mag = np.linalg.norm(U, axis=1)          # (M,)

        # --- build a separate deformation heatmap window ---
        heat_bgr = make_disp_heatmap(
            (frame.shape[0], frame.shape[1]),
            X0_in,              # use reference positions as anchors
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
