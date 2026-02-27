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
# All sensitive values live ONLY in .streamlit/secrets.toml
# Access via st.secrets["section"]["key"]
#
# Required sections:
#   [gmail]        → user, app_password, subject_prefix
#   [firebase]     → database_url, db_node
#   [firebase_key] → all service-account JSON fields

def _secret(section: str, key: str, fallback=None):
    """Safe helper — returns fallback instead of crashing if key is missing."""
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError):
        return fallback

# Gmail (IMAP)
GMAIL_USER           = _secret("gmail", "user",           "")
GMAIL_APP_PASSWORD   = _secret("gmail", "app_password",   "")
GMAIL_SUBJECT_PREFIX = _secret("gmail", "subject_prefix", "TerrLeaf Image:")

# Firebase Realtime DB
FIREBASE_DB_URL  = _secret("firebase", "database_url", "")
FIREBASE_DB_NODE = _secret("firebase", "db_node",      "leaf_records")

# Gmail OAuth (file-based token — path only, no secret needed here)
GMAIL_SCOPES           = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_TOKEN_PATH       = "gmail_token.json"
GMAIL_CREDENTIALS_PATH = "gmail_oauth_credentials.json"

# Image cache directory
IMG_CACHE_DIR = ".terraleaf_img_cache"
os.makedirs(IMG_CACHE_DIR, exist_ok=True)

# ── Re-use theme constants from ui.py ─────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to Firebase …")
def _init_firebase():
    """
    Initialise Firebase Admin SDK once per process.
    Reads service-account key entirely from st.secrets["firebase_key"] —
    no JSON file on disk required.
    """
    if not FIREBASE_OK:
        return None, "firebase-admin not installed. Run: pip install firebase-admin"
    if not FIREBASE_DB_URL:
        return None, (
            "Firebase database_url missing in secrets.toml.\n\n"
            "Add it under [firebase] in .streamlit/secrets.toml."
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
            "auth_provider_x509_cert_url": key_sec.get("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url":        key_sec.get("client_x509_cert_url", ""),
        }
        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_info)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
        return rtdb, None
    except Exception as e:
        return None, str(e)



def fetch_all_records() -> tuple[list[dict], str | None]:
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


# ══════════════════════════════════════════════════════════════════════════════
# GMAIL IMAGE FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(record_id: str) -> str:
    return os.path.join(IMG_CACHE_DIR, f"{record_id}.jpg")


def _fetch_image_imap(record_id: str) -> Image.Image | None:
    """
    Fetch the image attachment for *record_id* via IMAP App-Password.
    Caches to disk so it is fetched only once.
    """
    cache = _cache_path(record_id)
    if os.path.exists(cache):
        return Image.open(cache).convert("RGB")

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        mail.select("inbox")

        subject_query = f'SUBJECT "{GMAIL_SUBJECT_PREFIX} {record_id}"'
        _, msg_ids = mail.search(None, subject_query)
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
            if ct in ("image/jpeg", "image/png", "image/webp"):
                img_bytes = part.get_payload(decode=True)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(cache, "JPEG")
                return img
    except Exception:
        pass
    return None


def _fetch_image_gmail_api(record_id: str) -> Image.Image | None:
    """
    Fetch via Gmail REST API (OAuth2).  Falls back silently if token missing.
    """
    if not GMAIL_API_OK:
        return None
    cache = _cache_path(record_id)
    if os.path.exists(cache):
        return Image.open(cache).convert("RGB")

    try:
        creds = None
        if os.path.exists(GMAIL_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, GMAIL_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                return None   # OAuth not configured yet
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


def get_leaf_image(record_id: str) -> Image.Image | None:
    """Try Gmail API first, fall back to IMAP App-Password."""
    img = _fetch_image_gmail_api(record_id)
    if img is None:
        img = _fetch_image_imap(record_id)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models …")
def _load_models():
    try:
        from cnn_prediction import load_models
        return load_models(), None
    except Exception as e:
        return None, str(e)


def run_prediction(img: Image.Image) -> dict | None:
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


def _metric_row(items: list[tuple[str, str]]):
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    # ── Header ────────────────────────────────────────────────────────────────
    try:
        ui.page_header("DATABASE DASHBOARD", "FIREBASE REALTIME DB")
    except Exception:
        pass
    st.markdown("<h1>🗄️ Database Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#a7d9a7;font-size:.97rem;margin-top:.1rem'>"
        "Records submitted via the <strong>TerraLeaf Android app</strong> — "
        "leaf images are fetched directly from Gmail and can be re-analysed "
        "by the on-device AI pipeline.</p>",
        unsafe_allow_html=True,
    )

    # ── Load records ──────────────────────────────────────────────────────────
    with st.spinner("Fetching records from Firebase …"):
        records, err = fetch_all_records()

    if err:
        st.error(f"**Firebase error:** {err}")
        with st.expander("Setup instructions"):
            st.markdown(
                "1. Go to [Firebase Console](https://console.firebase.google.com) "
                "→ **Project Settings** → **Service Accounts** → **Generate new private key**\n\n"
                "2. Open the downloaded JSON and paste each field under `[firebase_key]` "
                "in `.streamlit/secrets.toml` (see the ⚙️ Setup Guide at the bottom of this page)\n\n"
                "3. Also add `database_url` under `[firebase]` in `secrets.toml`\n\n"
                "4. Install: `pip install firebase-admin`"
            )
        return

    if not records:
        st.info("No records found in Firebase yet. Submit data via the Android app.")
        return

    # Convert to DataFrame for analytics
    df = pd.DataFrame(records)
    # Normalise numeric columns
    num_cols = [
        "soil_moisture", "soil_pH", "soil_temperature",
        "nitrogen", "phosphorus", "potassium",
        "mean_green_intensity", "color_variance",
        "texture_entropy", "spot_area_ratio", "disease_color_index",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True
        ).dt.tz_localize(None)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Total Records", f"{len(df):,}")
    k2.metric("📷 Images in Cache",
              str(len([f for f in os.listdir(IMG_CACHE_DIR) if f.endswith(".jpg")])))
    k3.metric("🌡️ Avg Soil Temp",
              f"{df['soil_temperature'].mean():.1f} °C" if "soil_temperature" in df.columns else "—")
    k4.metric("💧 Avg Moisture",
              f"{df['soil_moisture'].mean():.1f}%" if "soil_moisture" in df.columns else "—")

    st.markdown("---")

    # ── Analytics panels ──────────────────────────────────────────────────────
    _section("📊 Analytics")
    tab_soil, tab_time, tab_corr = st.tabs(["Soil Overview", "Timeline", "Correlations"])

    with tab_soil:
        avail = [c for c in num_cols if c in df.columns]
        if avail:
            col_box, col_radar = st.columns(2)
            with col_box:
                fig_box = px.box(
                    df.melt(value_vars=avail[:6], var_name="Feature", value_name="Value"),
                    x="Feature", y="Value", color="Feature",
                    color_discrete_sequence=["#052e16","#166534","#15803d",
                                              "#22c55e","#39ff6a","#86efac"],
                    title="Soil Feature Distributions",
                )
                fig_box.update_layout(**PLOTLY_BASE, showlegend=False, height=300)
                st.plotly_chart(fig_box, use_container_width=True)
            with col_radar:
                means = [df[c].mean() for c in avail[:6]]
                norm  = [(v - min(means)) / (max(means) - min(means) + 1e-9) for v in means]
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
                        radialaxis=dict(visible=True, range=[0,1],
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
                **{k: v for k, v in PLOTLY_BASE.items() if k not in ("xaxis","yaxis")},
                height=420,
            )
            st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")

    # ── Records table ─────────────────────────────────────────────────────────
    _section("📋 All Records")

    # Search/filter bar
    search = st.text_input("🔍 Search by Record ID or User UID", placeholder="LEAF_…")
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

    st.markdown("---")

    # ── Per-record detail + prediction ────────────────────────────────────────
    _section("🔬 Record Detail & Prediction")

    record_ids = [r.get("record_id", f"row_{i}") for i, r in enumerate(records)]
    selected_id = st.selectbox("Select a record to inspect", record_ids)
    rec = next((r for r in records if r.get("record_id") == selected_id), None)

    if rec is None:
        st.warning("Record not found.")
        return

    col_meta, col_img = st.columns([2, 1])

    with col_meta:
        _section("Metadata")
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:.8rem;"
            f"color:#a7d9a7;line-height:2'>"
            f"<b>Record ID:</b> {rec.get('record_id','—')}<br>"
            f"<b>Timestamp:</b> {_ts(rec)}<br>"
            f"<b>Submitted by:</b> {rec.get('submitted_by','—')}<br>"
            f"<b>Image file:</b> {rec.get('image_filename','—')}"
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
            ("🔵 Spot Area",       _fmt(rec, "spot_area_ratio",      "{:.4f}")),
            ("🩺 Disease Color",   _fmt(rec, "disease_color_index",  "{:.4f}")),
        ])

    with col_img:
        _section("Leaf Image (from Gmail)")
        with st.spinner("Fetching image from Gmail …"):
            img = get_leaf_image(selected_id)
        if img:
            st.image(img, caption=rec.get("image_filename", selected_id),
                     use_container_width=True)
        else:
            st.markdown(
                "<div style='background:rgba(57,255,106,0.03);border:1px dashed "
                "rgba(57,255,106,0.2);border-radius:8px;padding:2rem;text-align:center;"
                "color:#547a54;font-size:.8rem'>📷 Image not yet available in Gmail.<br>"
                "It may still be in transit.</div>",
                unsafe_allow_html=True,
            )
        # Manual image upload fallback
        uploaded = st.file_uploader(
            "Or upload the image manually", type=["png","jpg","jpeg","webp"],
            key=f"upload_{selected_id}",
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded manually", use_container_width=True)

    # ── AI Prediction ─────────────────────────────────────────────────────────
    st.markdown("---")
    _section("🤖 AI Disease Prediction")

    if img is None:
        st.info("Upload or fetch the leaf image above to enable prediction.")
        return

    run_pred = st.button("🔬 Run Prediction on this Leaf", type="primary",
                         key=f"pred_{selected_id}")
    pred_slot = st.empty()

    if run_pred or st.session_state.get(f"pred_result_{selected_id}"):
        if run_pred:
            with st.spinner("Running AI prediction …"):
                result = run_prediction(img)
            if result:
                st.session_state[f"pred_result_{selected_id}"] = result
        else:
            result = st.session_state.get(f"pred_result_{selected_id}")

        if result:
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
                g1, g2, g3 = st.columns(3)
                g1.plotly_chart(
                    ui.gauge(imf["mean_green_intensity"], "GREEN INTENSITY", 0, 255, "#39ff6a"),
                    use_container_width=True,
                )
                g2.plotly_chart(
                    ui.gauge(imf["spot_area_ratio"] * 100, "SPOT AREA %", 0, 100, "#fbbf24"),
                    use_container_width=True,
                )
                g3.plotly_chart(
                    ui.gauge(min(imf["disease_color_index"] * 20, 100),
                             "DISEASE COLOR INDEX", 0, 100, "#f87171"),
                    use_container_width=True,
                )
            except Exception:
                pass

            # Soil prediction table vs actual
            _section("Predicted vs Recorded Soil Conditions")
            sp = result["soil_predictions"]
            rows = []
            soil_keys = ["soil_moisture","soil_pH","soil_temperature",
                         "nitrogen","phosphorus","potassium"]
            for k in soil_keys:
                actual  = rec.get(k)
                pred_v  = sp.get(k)
                rows.append({
                    "Parameter": k,
                    "Recorded":  round(float(actual), 3) if actual is not None else "—",
                    "AI Prediction": round(float(pred_v), 3) if pred_v is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Parameter"),
                         use_container_width=True)

            # Recommendation
            _section("Recommendation")
            sev_colors_map = {
                "Healthy": "#14532d", "Mild": "#713f12",
                "Moderate": "#7c2d12", "Severe": "#450a0a",
            }
            bg = sev_colors_map.get(sev, "#1a2e1a")
            border = SEV_COLOR.get(sev, "#4ade80")
            st.markdown(
                f"<div style='background:{bg};border-left:4px solid {border};"
                f"border-radius:8px;padding:1rem 1.2rem;color:#f0fdf4;"
                f"font-size:.92rem;line-height:1.6'>"
                f"{RECS.get(sev,'')}</div>",
                unsafe_allow_html=True,
            )

        else:
            models_tuple, load_err = _load_models()
            if load_err:
                st.error(
                    f"**Could not load AI models:** {load_err}\n\n"
                    "Train the models first in the **🏋️ Train Models** tab."
                )
            else:
                st.error("Prediction failed. Check that the image is a valid leaf photo.")

    # ── Bulk predict all records with cached images ────────────────────────────
    st.markdown("---")
    _section("⚡ Bulk Prediction (all cached images)")

    cached_ids = [
        f.replace(".jpg", "")
        for f in os.listdir(IMG_CACHE_DIR) if f.endswith(".jpg")
    ]
    st.write(
        f"{len(cached_ids)} image(s) available in local cache out of "
        f"{len(records)} total records."
    )

    if st.button("🚀 Predict all cached images", key="bulk_pred"):
        models_tuple, load_err = _load_models()
        if load_err:
            st.error(f"Cannot load models: {load_err}")
        else:
            bulk_results = []
            prog = st.progress(0)
            for i, cid in enumerate(cached_ids):
                try:
                    img_c = Image.open(_cache_path(cid)).convert("RGB")
                    res = run_prediction(img_c)
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
                prog.progress((i + 1) / max(len(cached_ids), 1))

            prog.empty()
            if bulk_results:
                df_bulk = pd.DataFrame(bulk_results)
                st.success(f"✅ Predicted {len(df_bulk)} records.")
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
                       if k not in ("xaxis","yaxis")},
                    legend=dict(font=dict(color="#a7d9a7")),
                )
                fig_pie.update_traces(
                    marker=dict(line=dict(color="#050c05", width=2))
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No predictions could be made.")

    # ── Gmail OAuth setup wizard ──────────────────────────────────────────────
    with st.expander("⚙️ Gmail / Firebase Setup Guide"):
        st.markdown("""
### 🔐 All credentials go in `.streamlit/secrets.toml`
No passwords or keys are ever stored in the Python source files.

---

### Step 1 — Create the secrets file

Create this folder and file next to `main.py`:
```
your_project/
├── main.py
├── database_dashboard.py
├── .streamlit/
│   └── secrets.toml       ← put ALL credentials here
└── .gitignore             ← make sure secrets.toml is listed here!
```

---

### Step 2 — Firebase service-account key

1. Open [Firebase Console](https://console.firebase.google.com) → your project
2. **Project Settings** → **Service Accounts** → **Generate new private key**
3. Open the downloaded `.json` file and **paste each field** into `secrets.toml`:

```toml
[firebase]
database_url = "https://terraleaf-iot-default-rtdb.asia-southeast1.firebasedatabase.app"
db_node      = "leaf_records"

[firebase_key]
type                        = "service_account"
project_id                  = "terraleaf-iot"
private_key_id              = "abc123..."
private_key                 = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
client_email                = "firebase-adminsdk-xxx@terraleaf-iot.iam.gserviceaccount.com"
client_id                   = "1234567890"
auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
token_uri                   = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url        = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```
> ⚠️ For `private_key`, keep it as one line with `\n` for newlines (not real line breaks).

---

### Step 3 — Gmail App Password

```toml
[gmail]
user           = "terraleaf.Iot@gmail.com"
app_password   = "cbgo onmr kwdl gvpc"
subject_prefix = "TerrLeaf Image:"
```

---

### Step 4 — Protect your secrets

Add this to your `.gitignore`:
```
.streamlit/secrets.toml
firebase_key.json
gmail_token.json
```

---

### Step 5 — Install dependencies
```bash
pip install firebase-admin google-api-python-client google-auth-oauthlib google-auth-httplib2
```
        """)