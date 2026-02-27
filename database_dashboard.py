"""
database_dashboard.py  –  TerraLeaf Firebase Dashboard
═══════════════════════════════════════════════════════
Connects to the SAME Firebase Realtime Database used by the Android app:
  https://terraleaf-iot-default-rtdb.firebaseio.com  →  /leaf_records

Features:
  • Live records from Firebase with full soil & image metadata
  • Leaf images fetched from Gmail inbox (IMAP App-Password)
    Subject format: "TerrLeaf Image: {record_id}"
  • Per-record AI prediction (GradientBoosting, no GPU needed)
  • Bulk prediction over all locally-cached images
  • Analytics: soil distributions, time series, correlation heatmap
  • Gallery grid view — each record card shows its leaf image
  • Record detail panel: all fields + prediction results

All credentials come from .streamlit/secrets.toml — nothing is hardcoded.
"""

import os
import io
import base64
import json
import email as email_lib
import imaplib
import time
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, db as rtdb
    FIREBASE_OK = True
except ImportError:
    FIREBASE_OK = False

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    GMAIL_API_OK = True
except ImportError:
    GMAIL_API_OK = False

# ── Load credentials from .streamlit/secrets.toml ────────────────────────────
def _secret(section: str, key: str, fallback=None):
    """Safe helper — returns fallback instead of crashing if key is missing."""
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError, Exception):
        return fallback

# Gmail (IMAP)
GMAIL_USER           = _secret("gmail", "user",           "")
GMAIL_APP_PASSWORD   = _secret("gmail", "app_password",   "")
GMAIL_SUBJECT_PREFIX = _secret("gmail", "subject_prefix", "TerrLeaf Image:")

# Firebase Realtime DB
FIREBASE_DB_URL  = _secret("firebase", "database_url", "https://terraleaf-iot-default-rtdb.asia-southeast1.firebasedatabase.app")
FIREBASE_DB_NODE = _secret("firebase", "db_node",      "leaf_records")

# Gmail OAuth paths (optional)
GMAIL_SCOPES           = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_TOKEN_PATH       = "gmail_token.json"
GMAIL_CREDENTIALS_PATH = "gmail_oauth_credentials.json"

# Image cache directory
IMG_CACHE_DIR = ".terraleaf_img_cache"
os.makedirs(IMG_CACHE_DIR, exist_ok=True)

# ── Theme constants ───────────────────────────────────────────────────────────
try:
    import ui
    SEV_COLOR    = ui.SEV_COLOR
    GREEN_SCALE  = ui.GREEN_SCALE
    PLOTLY_BASE  = ui.PLOTLY_BASE
except ImportError:
    SEV_COLOR   = {"Healthy": "#4ade80", "Mild": "#facc15",
                   "Moderate": "#fb923c", "Severe": "#f87171"}
    GREEN_SCALE = [[0.0, "#052e16"], [0.5, "#16a34a"], [1.0, "#fde047"]]
    PLOTLY_BASE = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0fdf4"), margin=dict(t=40, b=20, l=10, r=10),
    )

RECS = {
    "Healthy":  "✅ Leaf looks healthy! Maintain current irrigation and fertilisation schedule.",
    "Mild":     "⚠️ Early signs detected. Increase potassium supply and reduce leaf wetness.",
    "Moderate": "🔶 Moderate disease. Apply targeted fungicide/pesticide and adjust soil pH.",
    "Severe":   "🚨 Severe infection. Isolate the plant immediately and consult an agronomist.",
}

SOIL_COLS = [
    "soil_moisture", "soil_pH", "soil_temperature",
    "nitrogen", "phosphorus", "potassium",
    "mean_green_intensity", "color_variance",
    "texture_entropy", "spot_area_ratio", "disease_color_index",
]


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE  –  same project / same DB used by the Android app
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to Firebase …")
def _init_firebase():
    """
    Initialise Firebase Admin SDK once per process.
    Reads the service-account key entirely from st.secrets["firebase_key"].
    Uses the same Firebase project as the Android app:
        terraleaf-iot  →  terraleaf-iot-default-rtdb.firebaseio.com
    """
    if not FIREBASE_OK:
        return None, "firebase-admin not installed. Run: pip install firebase-admin"
    if not FIREBASE_DB_URL:
        return None, (
            "Firebase database_url missing in secrets.toml.\n\n"
            "Add it under [firebase] in .streamlit/secrets.toml.\n"
            "Expected value: https://terraleaf-iot-default-rtdb.firebaseio.com"
        )
    try:
        key_sec = st.secrets.get("firebase_key", {})
        if not key_sec or not key_sec.get("private_key"):
            return None, (
                "Firebase service-account key missing in secrets.toml.\n\n"
                "Paste all fields from your downloaded JSON under "
                "[firebase_key] in .streamlit/secrets.toml."
            )
        sa_info = {
            "type":                        key_sec.get("type", "service_account"),
            "project_id":                  key_sec["project_id"],
            "private_key_id":              key_sec["private_key_id"],
            "private_key":                 key_sec["private_key"].replace("\\n", "\n"),
            "client_email":                key_sec["client_email"],
            "client_id":                   key_sec["client_id"],
            "auth_uri":                    key_sec.get("auth_uri",  "https://accounts.google.com/o/oauth2/auth"),
            "token_uri":                   key_sec.get("token_uri", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": key_sec.get("auth_provider_x509_cert_url",
                                                        "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url":        key_sec.get("client_x509_cert_url", ""),
        }
        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_info)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
        return rtdb, None
    except Exception as e:
        return None, str(e)


def fetch_all_records() -> tuple:
    """Pull every child from /leaf_records. Returns (records_list, error_msg)."""
    rtdb_ref, err = _init_firebase()
    if err:
        return [], err
    try:
        snap = rtdb_ref.reference(FIREBASE_DB_NODE).get()
        if not snap:
            return [], None
        records = list(snap.values()) if isinstance(snap, dict) else []
        return records, None
    except Exception as e:
        return [], str(e)


def fetch_single_record(record_id: str) -> tuple:
    """Fetch a single record by ID. Returns (record_dict, error_msg)."""
    rtdb_ref, err = _init_firebase()
    if err:
        return None, err
    try:
        snap = rtdb_ref.reference(f"{FIREBASE_DB_NODE}/{record_id}").get()
        return snap, None
    except Exception as e:
        return None, str(e)


def delete_record(record_id: str) -> str | None:
    """Delete a record from Firebase. Returns error string or None on success."""
    rtdb_ref, err = _init_firebase()
    if err:
        return err
    try:
        rtdb_ref.reference(f"{FIREBASE_DB_NODE}/{record_id}").delete()
        # Also remove cached image
        cache = _cache_path(record_id)
        if os.path.exists(cache):
            os.remove(cache)
        return None
    except Exception as e:
        return str(e)


# ══════════════════════════════════════════════════════════════════════════════
# GMAIL IMAGE FETCHER
# The Android app (DataEntryActivity.java) sends images via SMTP to
# terraleaf.Iot@gmail.com with subject: "TerrLeaf Image: {record_id}"
# We fetch them back here via IMAP with the same App-Password from secrets.toml
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(record_id: str) -> str:
    return os.path.join(IMG_CACHE_DIR, f"{record_id}.jpg")


def _fetch_image_imap(record_id: str) -> "Image.Image | None":
    """
    Fetch the leaf image for record_id via IMAP App-Password.
    Matches the subject sent by DataEntryActivity:  "TerrLeaf Image: {record_id}"
    Caches to disk so each image is only downloaded once.
    """
    cache = _cache_path(record_id)
    if os.path.exists(cache):
        try:
            return Image.open(cache).convert("RGB")
        except Exception:
            os.remove(cache)  # corrupt cache — re-fetch

    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        return None

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        mail.select("inbox")

        # Try exact subject match first (most reliable)
        exact_subject = f'{GMAIL_SUBJECT_PREFIX} {record_id}'
        _, msg_ids = mail.search(None, f'SUBJECT "{exact_subject}"')
        ids = msg_ids[0].split()

        # Fallback: search just by record_id in subject
        if not ids:
            _, msg_ids = mail.search(None, f'SUBJECT "{record_id}"')
            ids = msg_ids[0].split()

        if not ids:
            mail.logout()
            return None

        # Take the most recent matching email
        _, raw = mail.fetch(ids[-1], "(RFC822)")
        mail.logout()

        msg = email_lib.message_from_bytes(raw[0][1])
        for part in msg.walk():
            ct = part.get_content_type()
            fname = part.get_filename() or ""
            if ct in ("image/jpeg", "image/png", "image/webp") or fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_bytes = part.get_payload(decode=True)
                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(cache, "JPEG", quality=92)
                    return img
    except Exception as e:
        pass  # silent fail — image just won't show
    return None


def _fetch_image_gmail_api(record_id: str) -> "Image.Image | None":
    """Fetch via Gmail REST API (OAuth2). Falls back silently if token missing."""
    if not GMAIL_API_OK:
        return None
    cache = _cache_path(record_id)
    if os.path.exists(cache):
        try:
            return Image.open(cache).convert("RGB")
        except Exception:
            pass

    try:
        creds = None
        if os.path.exists(GMAIL_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, GMAIL_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                return None
            with open(GMAIL_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())

        service = build("gmail", "v1", credentials=creds)
        query   = f'subject:"{GMAIL_SUBJECT_PREFIX} {record_id}"'
        result  = service.users().messages().list(userId="me", q=query, maxResults=1).execute()
        msgs    = result.get("messages", [])
        if not msgs:
            return None

        msg = service.users().messages().get(
            userId="me", id=msgs[0]["id"], format="full"
        ).execute()

        for part in msg["payload"].get("parts", []):
            mime = part.get("mimeType", "")
            if mime.startswith("image/"):
                att_id = part["body"].get("attachmentId")
                if att_id:
                    att = service.users().messages().attachments().get(
                        userId="me", messageId=msgs[0]["id"], id=att_id
                    ).execute()
                    img_bytes = base64.urlsafe_b64decode(att["data"])
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(cache, "JPEG")
                    return img
    except Exception:
        pass
    return None


def get_leaf_image(record_id: str) -> "Image.Image | None":
    """Try Gmail API first, fall back to IMAP App-Password."""
    img = _fetch_image_gmail_api(record_id)
    if img is None:
        img = _fetch_image_imap(record_id)
    return img


def cached_image_ids() -> list:
    """Return list of record_ids that have cached images on disk."""
    return [
        f.replace(".jpg", "")
        for f in os.listdir(IMG_CACHE_DIR) if f.endswith(".jpg")
    ]


# ══════════════════════════════════════════════════════════════════════════════
# AI PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models …")
def _load_models():
    try:
        from cnn_prediction import load_models
        return load_models(), None
    except FileNotFoundError:
        return None, "Model files not found. Train models first in the 🏋️ Train tab."
    except Exception as e:
        return None, str(e)


def run_prediction(img: "Image.Image") -> "dict | None":
    models, err = _load_models()
    if err or models is None:
        return None
    clf, scaler, le, regressors = models
    try:
        from cnn_prediction import predict
        return predict(img, clf, scaler, le, regressors)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str):
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:.65rem;"
        f"color:#547a54;letter-spacing:.14em;text-transform:uppercase;"
        f"margin:1.4rem 0 .5rem;border-bottom:1px solid rgba(74,222,128,.1);"
        f"padding-bottom:.25rem'>{title}</div>",
        unsafe_allow_html=True,
    )


def _metric_row(items: list):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)


def _sev_badge(sev: str):
    color = SEV_COLOR.get(sev, "#888")
    st.markdown(
        f"<span style='background:{color}1a;color:{color};"
        f"border:1px solid {color}55;border-radius:4px;"
        f"padding:3px 12px;font-size:.75rem;font-weight:700;"
        f"font-family:JetBrains Mono,monospace;text-transform:uppercase'>"
        f"{sev}</span>",
        unsafe_allow_html=True,
    )


def _fmt(rec: dict, key: str, fmt: str = "{:.3f}", default: str = "—") -> str:
    v = rec.get(key)
    if v is None:
        return default
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def _ts(rec: dict) -> str:
    ts = rec.get("timestamp")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts / 1000).strftime("%d %b %Y  %H:%M:%S")
    return "—"


def _img_thumb(record_id: str, width: int = 120) -> str:
    """Return an <img> HTML tag with base64-encoded thumbnail or a placeholder."""
    cache = _cache_path(record_id)
    if os.path.exists(cache):
        try:
            img = Image.open(cache).convert("RGB")
            img.thumbnail((width, width))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return (
                f"<img src='data:image/jpeg;base64,{b64}' "
                f"style='width:{width}px;height:{width}px;object-fit:cover;"
                f"border-radius:6px;border:1px solid rgba(74,222,128,0.2)'/>"
            )
        except Exception:
            pass
    return (
        f"<div style='width:{width}px;height:{width}px;border-radius:6px;"
        f"border:1px dashed rgba(74,222,128,0.2);display:flex;"
        f"align-items:center;justify-content:center;"
        f"color:#547a54;font-size:.7rem;text-align:center'>📷<br>No image</div>"
    )


def _severity_color_bg(sev: str) -> str:
    m = {"Healthy": "#14532d", "Mild": "#713f12", "Moderate": "#7c2d12", "Severe": "#450a0a"}
    return m.get(sev, "#1a2e1a")


def _pred_result_card(result: dict):
    """Render a full prediction result panel."""
    sev = result["severity"]

    c1, c2, c3 = st.columns(3)
    c1.metric("🩺 Disease Detected", result["disease_type"])
    c2.metric("📊 Confidence",        f"{result['confidence']:.1f}%")
    c3.metric("💚 Health Score",       f"{result['health_score']:.1f} / 100")

    st.markdown("<br>", unsafe_allow_html=True)
    _sev_badge(sev)
    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bar chart
    df_cp = (
        pd.DataFrame({
            "Disease":         list(result["class_probs"].keys()),
            "Probability (%)": list(result["class_probs"].values()),
        })
        .sort_values("Probability (%)", ascending=False)
    )
    fig_bar = px.bar(
        df_cp, x="Disease", y="Probability (%)",
        color="Probability (%)", color_continuous_scale=GREEN_SCALE,
        title="Disease Probability Distribution",
    )
    fig_bar.update_layout(**PLOTLY_BASE, coloraxis_showscale=False, height=260)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Gauges
    imf = result["image_features"]
    try:
        import ui as _ui
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(
            _ui.gauge(imf["mean_green_intensity"], "GREEN INTENSITY", 0, 255, "#39ff6a"),
            use_container_width=True,
        )
        g2.plotly_chart(
            _ui.gauge(imf["spot_area_ratio"] * 100, "SPOT AREA %", 0, 100, "#fbbf24"),
            use_container_width=True,
        )
        g3.plotly_chart(
            _ui.gauge(min(imf["disease_color_index"] * 20, 100), "DISEASE COLOR INDEX", 0, 100, "#f87171"),
            use_container_width=True,
        )
    except Exception:
        pass

    # Soil prediction table
    _section("Predicted vs Recorded Soil Conditions")


def _render_soil_pred_table(rec: dict, soil_preds: dict):
    rows = []
    for k in ["soil_moisture", "soil_pH", "soil_temperature", "nitrogen", "phosphorus", "potassium"]:
        actual = rec.get(k)
        pred_v = soil_preds.get(k)
        rows.append({
            "Parameter":    k,
            "Recorded":     round(float(actual), 3) if actual is not None else "—",
            "AI Prediction": round(float(pred_v), 3) if pred_v is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Parameter"), use_container_width=True)


def _render_recommendation(sev: str):
    _section("Recommendation")
    bg     = _severity_color_bg(sev)
    border = SEV_COLOR.get(sev, "#4ade80")
    st.markdown(
        f"<div style='background:{bg};border-left:4px solid {border};"
        f"border-radius:8px;padding:1rem 1.2rem;color:#f0fdf4;"
        f"font-size:.92rem;line-height:1.6'>"
        f"{RECS.get(sev, '')}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_overview(df: pd.DataFrame):
    """KPIs + analytics charts."""
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📋 Total Records",    f"{len(df):,}")
    k2.metric("📷 Images Cached",    str(len(cached_image_ids())))
    k3.metric("🌡️ Avg Soil Temp",
              f"{df['soil_temperature'].mean():.1f} °C" if "soil_temperature" in df.columns else "—")
    k4.metric("💧 Avg Moisture",
              f"{df['soil_moisture'].mean():.1f}%" if "soil_moisture" in df.columns else "—")
    k5.metric("🧪 Avg pH",
              f"{df['soil_pH'].mean():.2f}" if "soil_pH" in df.columns else "—")

    st.markdown("---")
    _section("📊 Analytics")
    tab_soil, tab_time, tab_corr = st.tabs(["Soil Overview", "Timeline", "Correlations"])

    with tab_soil:
        avail = [c for c in SOIL_COLS if c in df.columns]
        if avail:
            col_box, col_radar = st.columns(2)
            with col_box:
                fig_box = px.box(
                    df.melt(value_vars=avail[:6], var_name="Feature", value_name="Value"),
                    x="Feature", y="Value", color="Feature",
                    color_discrete_sequence=["#052e16", "#166534", "#15803d",
                                             "#22c55e", "#39ff6a", "#86efac"],
                    title="Soil Feature Distributions",
                )
                fig_box.update_layout(**PLOTLY_BASE, showlegend=False, height=300)
                st.plotly_chart(fig_box, use_container_width=True)
            with col_radar:
                means = [df[c].mean() for c in avail[:6]]
                mn, mx = min(means), max(means)
                norm = [(v - mn) / (mx - mn + 1e-9) for v in means]
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=norm + [norm[0]], theta=avail[:6] + [avail[0]],
                    fill="toself",
                    line=dict(color="#39ff6a", width=2),
                    fillcolor="rgba(57,255,106,0.10)",
                ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor="rgba(9,19,10,0.6)",
                        radialaxis=dict(visible=True, range=[0, 1],
                                        gridcolor="rgba(57,255,106,0.09)",
                                        tickfont=dict(color="#547a54", size=8),
                                        linecolor="rgba(57,255,106,0.1)"),
                        angularaxis=dict(tickfont=dict(color="#547a54", size=9),
                                         gridcolor="rgba(57,255,106,0.09)",
                                         linecolor="rgba(57,255,106,0.1)"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    title=dict(text="Mean Soil Profile", font=dict(color="#a7d9a7")),
                    height=300, showlegend=False,
                )
                st.plotly_chart(fig_r, use_container_width=True)

    with tab_time:
        if "datetime" in df.columns and df["datetime"].notna().any():
            df_t = df.sort_values("datetime")
            fig_t = px.scatter(
                df_t, x="datetime", y="soil_moisture",
                color="soil_pH", color_continuous_scale=GREEN_SCALE,
                title="Soil Moisture over Time (colour = pH)",
                hover_data=["record_id"] if "record_id" in df.columns else None,
            )
            fig_t.update_layout(**PLOTLY_BASE, height=300)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No timestamp data available.")

    with tab_corr:
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if len(nc) >= 2:
            fig_h = px.imshow(
                df[nc].corr(), color_continuous_scale=GREEN_SCALE,
                title="Feature Correlation Heatmap", aspect="auto",
            )
            fig_h.update_layout(
                **{k: v for k, v in PLOTLY_BASE.items() if k not in ("xaxis", "yaxis")},
                height=420,
            )
            st.plotly_chart(fig_h, use_container_width=True)


def _tab_records_table(df: pd.DataFrame, records: list):
    """Searchable table of all records."""
    _section("📋 All Records")

    search = st.text_input("🔍 Search by Record ID, User UID, or any field",
                           placeholder="LEAF_… or user UID")
    if search:
        mask = df.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df_view = df[mask]
    else:
        df_view = df

    display_cols = [c for c in
                    ["record_id", "datetime", "submitted_by", "soil_moisture",
                     "soil_pH", "soil_temperature", "nitrogen", "phosphorus",
                     "potassium", "image_filename"]
                    if c in df_view.columns]
    st.dataframe(df_view[display_cols].reset_index(drop=True), use_container_width=True)

    # CSV export
    csv = df_view[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "terraleaf_records.csv", "text/csv")


def _tab_gallery(records: list):
    """
    Gallery view: each record rendered as a card with its leaf image thumbnail.
    Images are fetched from Gmail (IMAP) and cached locally.
    """
    _section("🖼️ Leaf Image Gallery")

    # Options
    col_fetch, col_cols = st.columns([3, 1])
    with col_cols:
        ncols = st.selectbox("Columns", [2, 3, 4], index=1, key="gallery_cols")
    with col_fetch:
        fetch_all = st.button("📥 Fetch all images from Gmail", key="fetch_all_imgs",
                              help="Downloads images for all records that don't yet have a cached copy")

    if fetch_all:
        prog = st.progress(0)
        fetched = 0
        for i, rec in enumerate(records):
            rid = rec.get("record_id", "")
            if rid and not os.path.exists(_cache_path(rid)):
                img = get_leaf_image(rid)
                if img:
                    fetched += 1
            prog.progress((i + 1) / max(len(records), 1))
        prog.empty()
        st.success(f"Fetched {fetched} new images. Reload to see updated thumbnails.")
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Grid layout
    cols = st.columns(ncols)
    for i, rec in enumerate(records):
        rid   = rec.get("record_id", f"row_{i}")
        ts    = _ts(rec)
        ph    = _fmt(rec, "soil_pH", "{:.2f}")
        moist = _fmt(rec, "soil_moisture", "{:.1f}%")

        with cols[i % ncols]:
            # Fetch/show image
            cached = os.path.exists(_cache_path(rid))
            if cached:
                try:
                    img = Image.open(_cache_path(rid)).convert("RGB")
                    st.image(img, use_container_width=True,
                             caption=rec.get("image_filename", rid))
                except Exception:
                    st.markdown(_img_thumb(rid, 200), unsafe_allow_html=True)
            else:
                # Show placeholder and offer inline fetch
                st.markdown(
                    f"<div style='background:rgba(57,255,106,0.03);"
                    f"border:1px dashed rgba(57,255,106,0.2);border-radius:8px;"
                    f"padding:1.5rem;text-align:center;color:#547a54;"
                    f"font-size:.75rem;margin-bottom:.5rem'>"
                    f"📷 Image not cached<br>{rid}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"Fetch image", key=f"fetch_{rid}_{i}"):
                    with st.spinner("Fetching from Gmail …"):
                        img = get_leaf_image(rid)
                    if img:
                        st.image(img, use_container_width=True)
                        st.success("Image fetched!")
                        st.rerun()
                    else:
                        st.warning("Image not found in Gmail.")

            # Card info
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:.72rem;"
                f"color:#a7d9a7;line-height:1.7;margin-bottom:1rem;"
                f"border-left:2px solid rgba(74,222,128,0.25);padding-left:.6rem'>"
                f"<b>{rid}</b><br>"
                f"📅 {ts}<br>"
                f"🧪 pH: {ph}  💧 {moist}</div>",
                unsafe_allow_html=True,
            )


def _tab_detail(records: list):
    """
    Per-record detail panel: full metadata + leaf image + AI prediction.
    This mirrors the RecordDetailActivity in the Android app.
    """
    _section("🔬 Record Detail & AI Prediction")

    record_ids = [r.get("record_id", f"row_{i}") for i, r in enumerate(records)]
    selected_id = st.selectbox("Select a record to inspect", record_ids,
                                key="detail_select")
    rec = next((r for r in records if r.get("record_id") == selected_id), None)

    if rec is None:
        st.warning("Record not found.")
        return

    # ── Two-column layout: metadata left, image right ────────────────────────
    col_meta, col_img = st.columns([2, 1])

    with col_meta:
        _section("Metadata")
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:.82rem;"
            f"color:#a7d9a7;line-height:2.1'>"
            f"<b>Record ID:</b>   {rec.get('record_id','—')}<br>"
            f"<b>Timestamp:</b>   {_ts(rec)}<br>"
            f"<b>Submitted by:</b> {rec.get('submitted_by','—')}<br>"
            f"<b>Image file:</b>  {rec.get('image_filename','—')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        _section("Soil Readings")
        _metric_row([
            ("💧 Moisture",  _fmt(rec, "soil_moisture",    "{:.2f}%")),
            ("🧪 pH",        _fmt(rec, "soil_pH",          "{:.3f}")),
            ("🌡️ Temp",      _fmt(rec, "soil_temperature", "{:.2f}°C")),
        ])
        _metric_row([
            ("🟢 Nitrogen",   _fmt(rec, "nitrogen",   "{:.2f} mg/kg")),
            ("🟠 Phosphorus", _fmt(rec, "phosphorus", "{:.2f} mg/kg")),
            ("🟡 Potassium",  _fmt(rec, "potassium",  "{:.2f} mg/kg")),
        ])

        _section("Image-Derived Metrics")
        _metric_row([
            ("🟢 Green Intensity", _fmt(rec, "mean_green_intensity", "{:.2f}")),
            ("🎨 Color Variance",  _fmt(rec, "color_variance",       "{:.4f}")),
            ("📐 Texture Entropy", _fmt(rec, "texture_entropy",      "{:.4f}")),
        ])
        _metric_row([
            ("🔵 Spot Area",     _fmt(rec, "spot_area_ratio",     "{:.4f}")),
            ("🩺 Disease Color", _fmt(rec, "disease_color_index", "{:.4f}")),
        ])

    with col_img:
        _section("Leaf Image  (from Gmail)")

        # Try to get cached image; if not, fetch now
        img: "Image.Image | None" = None
        if os.path.exists(_cache_path(selected_id)):
            try:
                img = Image.open(_cache_path(selected_id)).convert("RGB")
            except Exception:
                pass

        if img is None:
            with st.spinner("Fetching image from Gmail …"):
                img = get_leaf_image(selected_id)

        if img:
            st.image(img, caption=rec.get("image_filename", selected_id),
                     use_container_width=True)
        else:
            st.markdown(
                "<div style='background:rgba(57,255,106,0.03);border:1px dashed "
                "rgba(57,255,106,0.2);border-radius:8px;padding:2.5rem;"
                "text-align:center;color:#547a54;font-size:.8rem'>"
                "📷 Image not yet available in Gmail.<br>"
                "It may still be in transit, or credentials may be missing.</div>",
                unsafe_allow_html=True,
            )

        # Manual upload fallback
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Or upload the image manually",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"upload_{selected_id}",
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            # Save to cache for future use
            img.save(_cache_path(selected_id), "JPEG", quality=92)
            st.image(img, caption="Uploaded manually", use_container_width=True)
            st.success("Image saved to local cache.")

    # ── AI Prediction ─────────────────────────────────────────────────────────
    st.markdown("---")
    _section("🤖 AI Disease Prediction")

    if img is None:
        st.info("💡 Upload or fetch the leaf image above to enable prediction.")
        return

    run_pred = st.button("🔬 Run Prediction on this Leaf", type="primary",
                         key=f"pred_{selected_id}")

    if run_pred or st.session_state.get(f"pred_result_{selected_id}"):
        if run_pred:
            with st.spinner("Running AI prediction …"):
                result = run_prediction(img)
            if result:
                st.session_state[f"pred_result_{selected_id}"] = result
        else:
            result = st.session_state.get(f"pred_result_{selected_id}")

        if result:
            _pred_result_card(result)
            _render_soil_pred_table(rec, result["soil_predictions"])
            _render_recommendation(result["severity"])
        else:
            models_tuple, load_err = _load_models()
            if load_err:
                st.error(
                    f"**Could not load AI models:** {load_err}\n\n"
                    "Train the models first in the **🏋️ Train Models** tab."
                )
            else:
                st.error("Prediction failed. Check that the image is a valid leaf photo.")


def _tab_bulk_predict(records: list):
    """Bulk predict all cached images and show aggregate results."""
    _section("⚡ Bulk Prediction")

    c_ids = cached_image_ids()
    st.write(
        f"**{len(c_ids)}** images in local cache out of **{len(records)}** total records."
    )

    if not c_ids:
        st.info("No cached images. Go to the Gallery tab and fetch images from Gmail first.")
        return

    if st.button("🚀 Predict all cached images", type="primary", key="bulk_pred"):
        models_tuple, load_err = _load_models()
        if load_err:
            st.error(f"Cannot load models: {load_err}")
        else:
            bulk_results = []
            prog = st.progress(0)
            status = st.empty()
            for i, cid in enumerate(c_ids):
                status.text(f"Predicting {cid} … ({i+1}/{len(c_ids)})")
                try:
                    img_c = Image.open(_cache_path(cid)).convert("RGB")
                    res   = run_prediction(img_c)
                    if res:
                        bulk_results.append({
                            "record_id":    cid,
                            "disease_type": res["disease_type"],
                            "confidence":   round(res["confidence"], 1),
                            "health_score": round(res["health_score"], 1),
                            "severity":     res["severity"],
                        })
                except Exception:
                    pass
                prog.progress((i + 1) / max(len(c_ids), 1))

            prog.empty()
            status.empty()

            if bulk_results:
                df_bulk = pd.DataFrame(bulk_results)
                st.success(f"✅ Predicted {len(df_bulk)} records.")

                # Summary KPIs
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Healthy",  int((df_bulk["severity"] == "Healthy").sum()))
                b2.metric("Mild",     int((df_bulk["severity"] == "Mild").sum()))
                b3.metric("Moderate", int((df_bulk["severity"] == "Moderate").sum()))
                b4.metric("Severe",   int((df_bulk["severity"] == "Severe").sum()))

                st.dataframe(df_bulk, use_container_width=True)

                # Severity pie
                vc = df_bulk["severity"].value_counts().reset_index()
                vc.columns = ["Severity", "Count"]
                fig_pie = px.pie(
                    vc, names="Severity", values="Count",
                    color="Severity",
                    color_discrete_map=SEV_COLOR,
                    hole=0.4, title="Severity Distribution (Bulk)",
                )
                fig_pie.update_layout(
                    **{k: v for k, v in PLOTLY_BASE.items()
                       if k not in ("xaxis", "yaxis")},
                    legend=dict(font=dict(color="#a7d9a7")),
                )
                fig_pie.update_traces(marker=dict(line=dict(color="#050c05", width=2)))
                st.plotly_chart(fig_pie, use_container_width=True)

                # Health score histogram
                fig_hist = px.histogram(
                    df_bulk, x="health_score", nbins=20,
                    color_discrete_sequence=["#39ff6a"],
                    title="Health Score Distribution",
                )
                fig_hist.update_layout(**PLOTLY_BASE, height=280)
                st.plotly_chart(fig_hist, use_container_width=True)

                # CSV download
                csv = df_bulk.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download bulk results CSV", csv,
                                   "bulk_predictions.csv", "text/csv")
            else:
                st.warning("No predictions could be made.")


def _tab_delete(records: list):
    """Delete a record from Firebase (mirrors RecordDetailActivity delete flow)."""
    _section("🗑️ Delete Record")

    st.warning(
        "⚠️ Deleting a record removes it permanently from the Firebase Realtime Database. "
        "The local image cache copy will also be removed."
    )

    record_ids = [r.get("record_id", f"row_{i}") for i, r in enumerate(records)]
    del_id = st.selectbox("Record to delete", record_ids, key="delete_select")

    rec = next((r for r in records if r.get("record_id") == del_id), {})
    if rec:
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:.8rem;"
            f"color:#fca5a5;background:#450a0a;padding:.8rem 1rem;"
            f"border-radius:8px;border:1px solid #f8717133;margin:.5rem 0'>"
            f"Record: {del_id}<br>"
            f"Submitted: {_ts(rec)}<br>"
            f"Image: {rec.get('image_filename','—')}</div>",
            unsafe_allow_html=True,
        )

    confirm = st.checkbox(f"I confirm I want to permanently delete **{del_id}**",
                          key="del_confirm")
    if st.button("🗑️ Delete Record", type="primary", disabled=not confirm, key="del_btn"):
        with st.spinner("Deleting …"):
            err = delete_record(del_id)
        if err:
            st.error(f"Delete failed: {err}")
        else:
            st.success(f"✅ Record {del_id} deleted from Firebase.")
            # Clear session state for this record
            for k in list(st.session_state.keys()):
                if del_id in k:
                    del st.session_state[k]
            st.rerun()


def _tab_setup():
    """Setup guide for Firebase + Gmail credentials."""
    st.markdown("""
### 🔐 All credentials live in `.streamlit/secrets.toml`

---

### Step 1 — Create the secrets file

```
your_project/
├── main.py
├── database_dashboard.py
├── .streamlit/
│   └── secrets.toml       ← put ALL credentials here
└── .gitignore             ← always add secrets.toml here!
```

---

### Step 2 — Firebase service-account key

1. Open [Firebase Console](https://console.firebase.google.com) → **terraleaf-iot**
2. **Project Settings** → **Service Accounts** → **Generate new private key**
3. Paste each field into `secrets.toml`:

```toml
[firebase]
database_url = "https://terraleaf-iot-default-rtdb.asia-southeast1.firebasedatabase.app"
db_node      = "leaf_records"

[firebase_key]
type                        = "service_account"
project_id                  = "terraleaf-iot"
private_key_id              = "abc123..."
private_key                 = "-----BEGIN PRIVATE KEY-----\\nMIIE..."
client_email                = "firebase-adminsdk-fbsvc@terraleaf-iot.iam.gserviceaccount.com"
client_id                   = "101609650994088269753"
auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
token_uri                   = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url        = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

> ⚠️ Keep `private_key` on one line, replacing real newlines with `\\n`.

---

### Step 3 — Gmail App Password (IMAP image fetching)

The Android app emails images to `terraleaf.Iot@gmail.com`.
This dashboard reads them back via IMAP.

```toml
[gmail]
user           = "terraleaf.Iot@gmail.com"
app_password   = "cbgo onmr kwdl gvpc"
subject_prefix = "TerrLeaf Image:"
```

---

### Step 4 — Install dependencies

```bash
pip install firebase-admin pillow streamlit pandas plotly
# Optional for Gmail OAuth:
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
```

---

### Step 5 — Protect your secrets

`.gitignore`:
```
.streamlit/secrets.toml
gmail_token.json
.terraleaf_img_cache/
```
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER  (called from main.py or run standalone)
# ══════════════════════════════════════════════════════════════════════════════

def render():
    # ── Header ────────────────────────────────────────────────────────────────
    try:
        import ui as _ui
        _ui.page_header("DATABASE DASHBOARD", "FIREBASE REALTIME DB")
    except Exception:
        pass

    st.markdown(
        "<h1 style='margin-bottom:.2rem'>🗄️ TerraLeaf Database Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#a7d9a7;font-size:.97rem;margin-top:.1rem'>"
        "Live records from <strong>terraleaf-iot</strong> Firebase Realtime DB  ·  "
        "Leaf images fetched from Gmail  ·  "
        "AI disease prediction via on-device scikit-learn pipeline.</p>",
        unsafe_allow_html=True,
    )

    # Connection status badge
    col_status, col_refresh = st.columns([5, 1])
    with col_refresh:
        if st.button("🔄 Refresh", key="db_refresh"):
            st.rerun()

    # ── Load records ──────────────────────────────────────────────────────────
    with st.spinner("Fetching records from Firebase …"):
        records, err = fetch_all_records()

    if err:
        with col_status:
            st.error(f"**Firebase connection failed:** {err}")
        st.markdown("---")
        with st.expander("⚙️ Setup Guide", expanded=True):
            _tab_setup()
        return

    with col_status:
        st.success(
            f"✅ Connected to **terraleaf-iot** Firebase  ·  "
            f"**{len(records)}** records in `/leaf_records`"
        )

    if not records:
        st.info("No records found yet. Submit data via the TerraLeaf Android app.")
        with st.expander("⚙️ Setup Guide"):
            _tab_setup()
        return

    # Build DataFrame
    df = pd.DataFrame(records)
    for c in SOIL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True
        ).dt.tz_localize(None)

    # ── Tab navigation ────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "📋 Records Table",
        "🖼️ Image Gallery",
        "🔬 Record Detail",
        "⚡ Bulk Predict",
        "🗑️ Delete",
        "⚙️ Setup",
    ])

    with tabs[0]:
        _tab_overview(df)
    with tabs[1]:
        _tab_records_table(df, records)
    with tabs[2]:
        _tab_gallery(records)
    with tabs[3]:
        _tab_detail(records)
    with tabs[4]:
        _tab_bulk_predict(records)
    with tabs[5]:
        _tab_delete(records)
    with tabs[6]:
        _tab_setup()


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="TerraLeaf Database Dashboard",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Apply theme if ui module available
    try:
        import ui as _ui
        _ui.apply_theme()
    except Exception:
        pass
    render()