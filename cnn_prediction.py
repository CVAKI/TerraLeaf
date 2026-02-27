"""
cnn_prediction.py  –  Pure scikit-learn replacement (Python 3.13 compatible)
──────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the original PyTorch version.
Public API is identical so main.py requires zero changes:
    train(data, imgdata)  →  (classifier, scaler, le, regressors)
    load_models()         →  (classifier, scaler, le, regressors)
    predict(img, ...)     →  result dict

Image features:
  Original used MobileNetV2 deep features.
  This version uses a rich 135-dim hand-crafted descriptor:
    • Spatial colour stats: mean + std of R,G,B in a 3x3 grid  (54 dims)
    • 64-bin luminance histogram                                (64 dims)
    • 5 disease-specific scalars (same formula as original)     ( 5 dims)
    • Quadrant texture: local std in 2x2 quadrants x 3 channels (12 dims)
  Classifier: GradientBoostingClassifier (strong, no GPU needed)
"""

import os, sys, pickle, warnings

# ── Block TensorFlow ──────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _mod in list(sys.modules.keys()):
    if 'tensorflow' in _mod or 'keras' in _mod:
        del sys.modules[_mod]
sys.modules['tensorflow'] = None
sys.modules['keras'] = None

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE        = (128, 128)
CLF_PATH        = 'clf_model.pkl'       # replaces cnn_model.pth
SCALER_PATH     = 'scaler.pkl'
ENCODER_PATH    = 'label_encoder.pkl'
REG_PATH        = 'regressors.pkl'
IMG_SCALER_PATH = 'img_scaler.pkl'

SOIL_FEATURES = [
    'soil_moisture', 'soil_pH', 'soil_temperature',
    'nitrogen', 'phosphorus', 'potassium',
    'mean_green_intensity', 'color_variance',
    'texture_entropy', 'spot_area_ratio', 'disease_color_index'
]
TARGET_CLASS = 'disease_type'


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _open_image(img_input) -> Image.Image:
    """Accept a file path string or a PIL Image."""
    if isinstance(img_input, str):
        return Image.open(img_input).convert('RGB')
    return img_input.convert('RGB')


def extract_image_features(img_input) -> dict:
    """5 disease-specific scalars — identical formula to original."""
    img = _open_image(img_input).resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))

    spot_mask = gray < gray.mean()
    dci = (float(r[spot_mask].mean() / (g[spot_mask].mean() + 1e-9))
           if spot_mask.any() else 0.0)

    return {
        'mean_green_intensity': float(g.mean()),
        'color_variance':       float(arr.var()),
        'texture_entropy':      entropy,
        'spot_area_ratio':      float(spot_mask.mean()),
        'disease_color_index':  dci,
    }


def _rich_image_vector(img_input) -> np.ndarray:
    """
    135-dimensional descriptor used internally by the classifier.

    [  0: 54]  Spatial colour stats – mean+std of R,G,B in 3×3 grid   (54)
    [ 54:118]  64-bin luminance histogram                              (64)
    [118:123]  5 disease scalars from extract_image_features            (5)
    [123:135]  Texture: local std in 2×2 quadrants × 3 channels       (12)
    Total = 135
    """
    img = _open_image(img_input).resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h, w = arr.shape[:2]

    # 3×3 spatial grid colour statistics (54 dims)
    spatial = []
    for ri in range(3):
        for ci in range(3):
            patch = arr[ri*h//3:(ri+1)*h//3, ci*w//3:(ci+1)*w//3, :]
            for c in range(3):
                spatial.append(patch[:, :, c].mean())
                spatial.append(patch[:, :, c].std())
    spatial = np.array(spatial, dtype=np.float32)   # 54

    # 64-bin luminance histogram (64 dims)
    gray = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
    lum_hist, _ = np.histogram(gray, bins=64, range=(0, 255))
    lum_hist = (lum_hist / (lum_hist.sum() + 1e-9)).astype(np.float32)  # 64

    # Disease scalars (5 dims)
    feats = extract_image_features(img_input)
    scalars = np.array(list(feats.values()), dtype=np.float32)           # 5

    # Quadrant texture std (12 dims)
    quad_std = []
    for ri in range(2):
        for ci in range(2):
            patch = arr[ri*h//2:(ri+1)*h//2, ci*w//2:(ci+1)*w//2, :]
            for c in range(3):
                quad_std.append(patch[:, :, c].std())
    quad_std = np.array(quad_std, dtype=np.float32)                      # 12

    return np.concatenate([spatial, lum_hist, scalars, quad_std])        # 135


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(data: pd.DataFrame, imgdata: dict):
    """
    Train the classifier + regressors and save all models to disk.
    Returns (classifier, scaler, le, regressors) — same shape as original.
    """
    print("Backend: scikit-learn (no PyTorch / no DLL issues)")

    # Labels
    le = LabelEncoder()
    y_cls = le.fit_transform(data[TARGET_CLASS].astype(str))
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    # Tabular scaler (soil features)
    scaler = StandardScaler()
    X_tab  = scaler.fit_transform(data[SOIL_FEATURES])

    # Extract rich image feature vectors
    img_vecs, valid_idx = [], []
    total = len(data)
    for i in range(total):
        fname = f'img_{(i % 1000) + 1:04d}.png'
        if fname in imgdata:
            img_vecs.append(_rich_image_vector(imgdata[fname]))
            valid_idx.append(i)
        if (i + 1) % 100 == 0:
            print(f"  Feature extraction: {i+1}/{total}")

    valid_idx = np.array(valid_idx)
    X_img     = np.vstack(img_vecs)                          # (N, 135)
    y_v       = y_cls[valid_idx]
    print(f"Images matched: {len(valid_idx)}")

    # Scale image features
    img_scaler = StandardScaler()
    X_img_sc   = img_scaler.fit_transform(X_img)

    # Combine image + tabular features for classification
    X_tab_v = X_tab[valid_idx]
    X_all   = np.hstack([X_img_sc, X_tab_v])                # (N, 146)

    # Train / val split
    idx = np.arange(len(valid_idx))
    tr_i, va_i = train_test_split(idx, test_size=0.15,
                                  random_state=42, stratify=y_v)

    # Classifier
    print("Training GradientBoosting classifier …")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    clf.fit(X_all[tr_i], y_v[tr_i])
    val_acc = accuracy_score(y_v[va_i], clf.predict(X_all[va_i]))
    print(f"  Classifier val_acc = {val_acc:.4f}")

    # Regressors (soil feature prediction from tabular data)
    regressors = {}
    for feat in SOIL_FEATURES:
        y_r = data[feat].values[valid_idx]
        reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tab_v[tr_i], y_r[tr_i])
        rmse = mean_squared_error(y_r[va_i], reg.predict(X_tab_v[va_i])) ** 0.5
        print(f"  {feat}: RMSE={rmse:.4f}")
        regressors[feat] = reg

    # Persist everything
    with open(CLF_PATH,        'wb') as f: pickle.dump(clf,        f)
    with open(SCALER_PATH,     'wb') as f: pickle.dump(scaler,     f)
    with open(ENCODER_PATH,    'wb') as f: pickle.dump(le,         f)
    with open(REG_PATH,        'wb') as f: pickle.dump(regressors, f)
    with open(IMG_SCALER_PATH, 'wb') as f: pickle.dump(img_scaler, f)
    print("All models saved.")

    # Attach img_scaler so predict() can use it without signature change
    clf._img_scaler = img_scaler
    return clf, scaler, le, regressors


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load saved models. Returns (classifier, scaler, le, regressors)."""
    with open(CLF_PATH,        'rb') as f: clf        = pickle.load(f)
    with open(SCALER_PATH,     'rb') as f: scaler     = pickle.load(f)
    with open(ENCODER_PATH,    'rb') as f: le         = pickle.load(f)
    with open(REG_PATH,        'rb') as f: regressors = pickle.load(f)
    with open(IMG_SCALER_PATH, 'rb') as f: img_scaler = pickle.load(f)
    # Stash on clf so predict() can retrieve it without changing main.py
    clf._img_scaler = img_scaler
    return clf, scaler, le, regressors


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def predict(img_input, clf, scaler, le, regressors) -> dict:
    """
    Full inference pipeline.
    Signature matches original: predict(img, cnn, scaler, le, regressors)
    The 'clf' argument replaces 'cnn' — main.py passes it transparently.
    """
    # Image feature vector
    img_vec    = _rich_image_vector(img_input).reshape(1, -1)
    img_vec_sc = clf._img_scaler.transform(img_vec)

    # Tabular proxy (image-derived scalars fill the soil columns)
    img_feats   = extract_image_features(img_input)
    default_row = {f: 0.0 for f in SOIL_FEATURES}
    default_row.update(img_feats)
    tab_vec = scaler.transform(pd.DataFrame([default_row])[SOIL_FEATURES])

    # Combined feature vector
    X_all = np.hstack([img_vec_sc, tab_vec])                # (1, 146)

    # Classify
    proba      = clf.predict_proba(X_all)[0]
    pred_idx   = int(np.argmax(proba))
    disease    = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx]) * 100
    class_probs = {le.inverse_transform([i])[0]: float(p) * 100
                   for i, p in enumerate(proba)}

    # Soil predictions
    soil_preds = {f: float(r.predict(tab_vec)[0]) for f, r in regressors.items()}

    # Health score — identical formula to original
    spot   = img_feats['spot_area_ratio']
    dci    = img_feats['disease_color_index']
    health = max(0.0, min(100.0, 100 - spot * 200 - (dci - 1) * 20))
    severity = ("Healthy"  if health >= 80 else
                "Mild"     if health >= 60 else
                "Moderate" if health >= 40 else "Severe")

    return {
        'disease_type':     disease,
        'confidence':       confidence,
        'class_probs':      class_probs,
        'soil_predictions': soil_preds,
        'image_features':   img_feats,
        'health_score':     health,
        'severity':         severity,
    }