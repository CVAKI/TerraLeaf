import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io, zipfile, os
import ui

st.set_page_config(
    page_title="TerraLeaf · Disease Analyser",
    page_icon="terraleaf_icon.png" if os.path.exists("terraleaf_icon.png") else "🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
ui.apply_theme()


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models …")
def load_all_models():
    from cnn_prediction import load_models
    return load_models()


@st.cache_resource(show_spinner="Loading dataset …")
def load_dataset():
    from ImportFiles import data, imgdata
    return data, imgdata


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    ui.sidebar_header()
    st.markdown("---")

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:.65rem;"
        "color:#547a54;letter-spacing:.14em;text-transform:uppercase;"
        "margin-bottom:.5rem'>How It Works</div>",
        unsafe_allow_html=True
    )
    st.info(
        "1. Upload a leaf photo\n"
        "2. CNN classifies the disease\n"
        "3. Regressors estimate soil conditions\n"
        "4. Dashboard shows full diagnosis"
    )

    st.markdown("---")
    mode = st.radio(
        "Navigate",
        ["🔍 Predict", "📊 Dataset Overview", "🏋️ Train Models", "🗄️ Database"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    ui.sidebar_footer()


# ── Constants ─────────────────────────────────────────────────────────────────
RECS = {
    "Healthy":  "✅ Leaf looks healthy! Maintain current irrigation and fertilisation schedule.",
    "Mild":     "⚠️ Early signs detected. Increase potassium supply and reduce leaf wetness.",
    "Moderate": "🔶 Moderate disease. Apply targeted fungicide/pesticide and adjust soil pH.",
    "Severe":   "🚨 Severe infection. Isolate the plant immediately and consult an agronomist.",
}

SEV_COLOR = ui.SEV_COLOR


# ── Single-image result panel ─────────────────────────────────────────────────
def render_single_result(img, result, label):
    sev = result["severity"]

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(img, caption=label, use_container_width=True)
    with col_info:
        ui.card("DETECTED DISEASE", result["disease_type"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence",   f"{result['confidence']:.1f}%")
        m2.metric("Health Score", f"{result['health_score']:.1f} / 100")
        with m3:
            st.markdown("<br>", unsafe_allow_html=True)
            ui.severity_badge(sev)

    ui.section_title("Disease Probability Distribution")
    df_cp = (
        pd.DataFrame({
            "Disease":         list(result["class_probs"].keys()),
            "Probability (%)": list(result["class_probs"].values()),
        })
        .sort_values("Probability (%)", ascending=False)
    )
    fig_bar = px.bar(
        df_cp, x="Disease", y="Probability (%)",
        color="Probability (%)", color_continuous_scale=ui.GREEN_SCALE
    )
    fig_bar.update_layout(**ui.PLOTLY_BASE, coloraxis_showscale=False, height=270)
    st.plotly_chart(fig_bar, use_container_width=True)

    ui.section_title("Image-Derived Features")
    imf = result["image_features"]
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(
        ui.gauge(imf["mean_green_intensity"], "GREEN INTENSITY", 0, 255, "#39ff6a"),
        use_container_width=True
    )
    g2.plotly_chart(
        ui.gauge(imf["spot_area_ratio"] * 100, "SPOT AREA %", 0, 100, "#fbbf24"),
        use_container_width=True
    )
    g3.plotly_chart(
        ui.gauge(min(imf["disease_color_index"] * 20, 100), "DISEASE COLOR INDEX", 0, 100, "#f87171"),
        use_container_width=True
    )

    ui.section_title("Predicted Soil Conditions")
    sp = result["soil_predictions"]
    col_t, col_r = st.columns(2)
    with col_t:
        df_soil = pd.DataFrame({
            "Parameter":       list(sp.keys()),
            "Predicted Value": [round(v, 3) for v in sp.values()],
        }).set_index("Parameter")
        st.dataframe(df_soil, use_container_width=True)
    with col_r:
        keys = list(sp.keys())
        vals = [abs(v) / (abs(v) + 1) for v in sp.values()]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=keys + [keys[0]],
            fill="toself",
            line=dict(color="#39ff6a", width=2),
            fillcolor="rgba(57,255,106,0.10)",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(9,19,10,0.6)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="rgba(57,255,106,0.09)",
                    tickfont=dict(color="#547a54", size=8),
                    linecolor="rgba(57,255,106,0.1)",
                ),
                angularaxis=dict(
                    tickfont=dict(color="#547a54", size=9),
                    gridcolor="rgba(57,255,106,0.09)",
                    linecolor="rgba(57,255,106,0.1)",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=285, showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    ui.section_title("Recommendation")
    ui.recommendation_box(RECS.get(sev, "No recommendation available."), sev)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Predict":
    MAX_IMAGES = 5000
    VALID_EXT  = {".png", ".jpg", ".jpeg", ".webp"}

    ui.page_header("PREDICT", "MAX BATCH: 5,000 IMAGES")
    st.markdown("<h1>Leaf Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#a7d9a7;font-size:.97rem;margin-top:.1rem'>"
        "Upload individual images <strong>or</strong> a ZIP folder — "
        "up to <strong>5,000 images</strong> per run.</p>",
        unsafe_allow_html=True
    )

    upload_mode = st.radio(
        "Upload mode",
        ["🖼️ Select images", "📦 Upload ZIP folder"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_files = []

    if upload_mode == "🖼️ Select images":
        raw = st.file_uploader(
            "📁 Select images — hold **Ctrl** / **⌘** to pick multiple",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if raw:
            if len(raw) > MAX_IMAGES:
                st.warning(f"⚠️ {len(raw)} files selected — only the first {MAX_IMAGES:,} will be processed.")
                raw = raw[:MAX_IMAGES]
            uploaded_files = [(f.name, f) for f in raw]

    else:
        zip_file = st.file_uploader(
            "📦 Upload a ZIP file containing leaf images (max 5,000 images inside)",
            type=["zip"],
        )
        if zip_file:
            with st.spinner("📂 Reading ZIP …"):
                try:
                    zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
                    img_entries = sorted([
                        e for e in zf.namelist()
                        if os.path.splitext(e.lower())[1] in VALID_EXT
                        and not os.path.basename(e).startswith(".")
                        and "__MACOSX" not in e
                    ])
                    total_found = len(img_entries)
                    if total_found == 0:
                        st.error("No supported images found inside the ZIP (PNG/JPG/JPEG/WEBP).")
                    else:
                        if total_found > MAX_IMAGES:
                            st.warning(
                                f"⚠️ ZIP contains {total_found:,} images — "
                                f"only the first {MAX_IMAGES:,} will be processed."
                            )
                            img_entries = img_entries[:MAX_IMAGES]
                        else:
                            st.success(f"✅ Found **{total_found:,}** images in ZIP.")
                        for entry in img_entries:
                            uploaded_files.append(
                                (os.path.basename(entry), io.BytesIO(zf.read(entry)))
                            )
                except Exception as e:
                    st.error(f"Failed to read ZIP: {e}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    if uploaded_files:
        n = len(uploaded_files)

        try:
            cnn, scaler, le, regressors = load_all_models()
            from cnn_prediction import predict
        except FileNotFoundError:
            st.error("⚠️ Models not found. Go to **🏋️ Train Models** first.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()

        st.info(f"🔬 Analysing **{n:,}** image{'s' if n > 1 else ''} …")
        prog_bar  = st.progress(0)
        prog_text = st.empty()
        CHUNK = max(1, min(50, n // 20 or 1))

        images, results, names = [], [], []
        errors = 0
        for idx, (nm, src) in enumerate(uploaded_files):
            try:
                img = Image.open(src).convert("RGB")
                res = predict(img, cnn, scaler, le, regressors)
            except Exception:
                img, res = None, None
                errors += 1
            images.append(img)
            results.append(res)
            names.append(nm)
            if (idx + 1) % CHUNK == 0 or idx == n - 1:
                prog_bar.progress((idx + 1) / n)
                prog_text.markdown(
                    f"**{idx+1:,} / {n:,}** analysed"
                    + (f" &nbsp;|&nbsp; ⚠️ {errors} error(s)" if errors else "")
                )

        prog_bar.empty()
        prog_text.empty()
        st.success(f"✅ Done — {n - errors:,} succeeded, {errors} failed.")

        # ── Single image ──────────────────────────────────────────────────────
        if n == 1:
            if results[0]:
                render_single_result(images[0], results[0], names[0])
            else:
                st.error("Analysis failed for this image.")

        # ── Batch mode ────────────────────────────────────────────────────────
        else:
            ui.section_title("📋 Batch Summary")

            summary_rows = []
            for nm, res in zip(names, results):
                if res:
                    summary_rows.append({
                        "File":         nm,
                        "Disease":      res["disease_type"],
                        "Confidence %": round(res["confidence"], 1),
                        "Health Score": round(res["health_score"], 1),
                        "Severity":     res["severity"],
                    })
                else:
                    summary_rows.append({
                        "File": nm, "Disease": "Error",
                        "Confidence %": None, "Health Score": None, "Severity": "Error",
                    })

            df_summary = pd.DataFrame(summary_rows)
            valid_res  = [r for r in results if r]

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Images", f"{n:,}")
            sc2.metric("Analysed OK",  f"{len(valid_res):,}")
            sc3.metric("Avg Health",   f"{df_summary['Health Score'].mean():.1f}" if valid_res else "—")
            sc4.metric("Errors",       str(errors))

            st.markdown("<br>", unsafe_allow_html=True)
            sev_filter = st.multiselect(
                "Filter by severity",
                options=["Healthy", "Mild", "Moderate", "Severe", "Error"],
                default=["Healthy", "Mild", "Moderate", "Severe", "Error"],
            )
            df_show = df_summary[df_summary["Severity"].isin(sev_filter)]
            st.dataframe(
                df_show, use_container_width=True, hide_index=True,
                height=min(420, 45 + len(df_show) * 38),
            )

            st.download_button(
                "⬇️ Download Full Results as CSV",
                data=df_summary.to_csv(index=False).encode(),
                file_name="terraleaf_analysis_results.csv",
                mime="text/csv",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            col_pie, col_bar = st.columns(2)
            with col_pie:
                sev_counts = df_summary["Severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig_sev = px.pie(
                    sev_counts, names="Severity", values="Count",
                    title="Severity Distribution",
                    color="Severity", color_discrete_map=SEV_COLOR, hole=0.42,
                )
                fig_sev.update_layout(
                    **{k: v for k, v in ui.PLOTLY_BASE.items() if k not in ("xaxis", "yaxis")},
                    height=310, legend=dict(font=dict(color="#a7d9a7")),
                )
                fig_sev.update_traces(marker=dict(line=dict(color="#050c05", width=2)))
                st.plotly_chart(fig_sev, use_container_width=True)
            with col_bar:
                fig_health = px.histogram(
                    df_summary[df_summary["Health Score"].notna()],
                    x="Health Score", color="Severity",
                    title="Health Score Distribution",
                    nbins=20, color_discrete_map=SEV_COLOR, barmode="stack",
                )
                fig_health.update_layout(**ui.PLOTLY_BASE, height=310)
                st.plotly_chart(fig_health, use_container_width=True)

            st.markdown("---")

            # Gallery
            ui.section_title("🖼️ Image Gallery")
            GALLERY_PAGE_SIZE = 40
            total_pages = max(1, (n + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
            page_num = (
                st.number_input(
                    f"Gallery page (1 – {total_pages})",
                    min_value=1, max_value=total_pages, value=1, step=1,
                )
                if total_pages > 1
                else 1
            )
            page_start = (page_num - 1) * GALLERY_PAGE_SIZE
            page_end   = min(page_start + GALLERY_PAGE_SIZE, n)
            page_imgs  = list(range(page_start, page_end))

            COLS_PER_ROW = 5
            for row_start in range(0, len(page_imgs), COLS_PER_ROW):
                cols = st.columns(COLS_PER_ROW)
                for ci, gi in enumerate(page_imgs[row_start:row_start + COLS_PER_ROW]):
                    res   = results[gi]
                    sev   = res["severity"] if res else "Error"
                    color = SEV_COLOR.get(sev, "#888")
                    with cols[ci]:
                        if images[gi]:
                            st.image(images[gi], use_container_width=True)
                        else:
                            st.markdown("❌")
                        st.markdown(
                            f"<div style='text-align:center;font-size:.68rem;color:#547a54;"
                            f"margin-top:-2px;word-break:break-all;"
                            f"font-family:JetBrains Mono,monospace;line-height:1.4'>"
                            f"{names[gi]}</div>"
                            f"<div style='text-align:center;margin-bottom:6px'>"
                            f"<span style='background:{color}1a;color:{color};"
                            f"border:1px solid {color}55;"
                            f"border-radius:4px;padding:2px 9px;font-size:.68rem;"
                            f"font-weight:600;font-family:JetBrains Mono,monospace;"
                            f"text-transform:uppercase'>{sev}</span></div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("---")

            # Detailed results
            ui.section_title("🔬 Detailed Results per Image")
            DETAIL_PAGE_SIZE = 20
            detail_pages = max(1, (n + DETAIL_PAGE_SIZE - 1) // DETAIL_PAGE_SIZE)
            detail_page = (
                st.number_input(
                    f"Detail page (1 – {detail_pages})",
                    min_value=1, max_value=detail_pages, value=1, step=1,
                    key="detail_page",
                )
                if detail_pages > 1
                else 1
            )
            d_start = (detail_page - 1) * DETAIL_PAGE_SIZE
            d_end   = min(d_start + DETAIL_PAGE_SIZE, n)

            for i in range(d_start, d_end):
                res   = results[i]
                label = (
                    f"📄 {names[i]}  —  {res['disease_type']}  ({res['severity']})"
                    if res
                    else f"📄 {names[i]}  —  Error"
                )
                with st.expander(label, expanded=(i == d_start)):
                    if res and images[i]:
                        render_single_result(images[i], res, names[i])
                    else:
                        st.error("Analysis failed for this image.")

    else:
        ui.upload_empty_state()


# ══════════════════════════════════════════════════════════════════════════════
# DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📊 Dataset Overview":
    ui.page_header("DATASET OVERVIEW")
    st.markdown("<h1>Dataset Overview</h1>", unsafe_allow_html=True)

    try:
        data, imgdata = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",    f"{len(data):,}")
    c2.metric("Features",      len(data.columns))
    c3.metric("Images Loaded", f"{len(imgdata):,}")

    ui.section_title("Sample Data")
    st.dataframe(data.head(20), use_container_width=True)

    if "disease_type" in data.columns:
        ui.section_title("Disease Distribution")
        vc = data["disease_type"].value_counts().reset_index()
        vc.columns = ["Disease", "Count"]
        col_p, col_b = st.columns(2)
        with col_p:
            green_seq = ["#052e16", "#166534", "#15803d", "#22c55e", "#39ff6a", "#86efac"]
            fig_pie = px.pie(
                vc, names="Disease", values="Count",
                color_discrete_sequence=green_seq, hole=0.42,
            )
            fig_pie.update_layout(
                **{k: v for k, v in ui.PLOTLY_BASE.items() if k not in ("xaxis", "yaxis")},
                legend=dict(font=dict(color="#a7d9a7")),
            )
            fig_pie.update_traces(marker=dict(line=dict(color="#050c05", width=2)))
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            fig_b = px.bar(
                vc, x="Disease", y="Count",
                color="Count", color_continuous_scale=ui.GREEN_SCALE,
            )
            fig_b.update_layout(**ui.PLOTLY_BASE, coloraxis_showscale=False)
            st.plotly_chart(fig_b, use_container_width=True)

    soil_cols = ["soil_moisture", "soil_pH", "soil_temperature",
                 "nitrogen", "phosphorus", "potassium"]
    avail = [c for c in soil_cols if c in data.columns]
    if avail:
        ui.section_title("Soil Feature Distributions")
        fig_box = px.box(
            data.melt(value_vars=avail, var_name="Feature", value_name="Value"),
            x="Feature", y="Value", color="Feature",
            color_discrete_sequence=["#052e16", "#166534", "#15803d", "#22c55e", "#39ff6a", "#86efac"],
        )
        fig_box.update_layout(**ui.PLOTLY_BASE, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        ui.section_title("Correlation Heatmap")
        fig_heat = px.imshow(
            data[num_cols].corr(),
            color_continuous_scale=ui.RG_HEATMAP_SCALE, aspect="auto",
        )
        fig_heat.update_layout(
            **{k: v for k, v in ui.PLOTLY_BASE.items() if k not in ("xaxis", "yaxis")},
            height=500,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🏋️ Train Models":
    ui.page_header("TRAIN MODELS")
    st.markdown("<h1>Train Models</h1>", unsafe_allow_html=True)

    ui.recommendation_box(
        "⚠️ Training will use the full dataset and may take several minutes depending on hardware.",
        "Moderate",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    epochs = st.slider("Max epochs", 5, 50, 20)

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Loading dataset …"):
            try:
                data, imgdata = load_dataset()
            except Exception as e:
                st.error(f"Dataset load failed: {e}")
                st.stop()

        prog = st.progress(0)
        log  = st.empty()
        log.info("Starting training pipeline …")
        try:
            from cnn_prediction import train
            prog.progress(10)
            cnn, scaler, le, regressors = train(data, imgdata)
            prog.progress(100)
            log.success("✅ Training complete! All models saved to disk.")
            st.balloons()
            load_all_models.clear()
        except Exception as e:
            st.error(f"Training failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🗄️ Database":
    import database_dashboard
    database_dashboard.render()