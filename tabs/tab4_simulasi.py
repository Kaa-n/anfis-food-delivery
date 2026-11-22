"""Tab 4: simulasi input dan inspeksi lima lapisan (Anggota 4)."""
import numpy as np
import pandas as pd
import streamlit as st
from tabs.components import (
    FEATURE_COLS, TRAFFIC_MAP, WEATHER_MAP, denormalize_output,
    normalize_single_input, render_layer_inspection,
)


def render_tab4_simulasi():
    st.subheader("Simulasi Skenario & Validasi Perhitungan Per Lapisan")
    if st.session_state.model is None or st.session_state.data_pack is None:
        st.warning(
            "Siapkan dataset di Tab 2 dan latih model di Tab 3.",
            icon=":material/warning:",
        )
        return

    model = st.session_state.model
    _, _, X_test, _, y_test, scaler_info = st.session_state.data_pack

    mode = st.radio(
        "Sumber input",
        ["Input manual", "Sampel data uji"],
        horizontal=True,
        key="tab4_source",
    )

    y_actual_sample = None
    if mode == "Sampel data uji":
        if len(X_test) == 0:
            st.warning("Data uji kosong.", icon=":material/warning:")
            return
        index = st.number_input(
            "Nomor sampel data uji",
            min_value=1,
            max_value=len(X_test),
            value=1,
            step=1,
            key="tab4_sample",
        )
        idx_zero = int(index) - 1
        x_norm_single = np.asarray(X_test[idx_zero]).reshape(1, -1)
        raw_vec = (
            x_norm_single[0] * (scaler_info["x_max"] - scaler_info["x_min"] + 1e-8)
            + scaler_info["x_min"]
        )
        y_actual_sample = float(denormalize_output(y_test[idx_zero], scaler_info))
        st.caption(
            "Sampel dan bobot model ini dapat dipakai untuk validasi hitung manual estafet."
        )
    else:
        c1, c2, c3 = st.columns(3)
        in_dist = c1.number_input(
            "Jarak Restoran - Pelanggan (km)", 0.1, 50.0, 6.5, 0.5, key="sim_dist"
        )
        in_prep = c2.number_input(
            "Waktu Persiapan Makanan (menit)", 1.0, 90.0, 15.0, 1.0, key="sim_prep"
        )
        in_exp = c3.number_input(
            "Pengalaman Kurir (tahun)", 0.0, 30.0, 3.0, 0.5, key="sim_exp"
        )
        c4, c5 = st.columns(2)
        traffic = c4.selectbox(
            "Tingkat Kemacetan Lalu Lintas", list(TRAFFIC_MAP), index=1, key="sim_traf"
        )
        weather = c5.selectbox("Kondisi Cuaca", list(WEATHER_MAP), key="sim_weat")
        raw_vec = [in_dist, in_prep, in_exp, TRAFFIC_MAP[traffic], WEATHER_MAP[weather]]
        x_norm_single = normalize_single_input(raw_vec, scaler_info).reshape(1, -1)

    if np.any((x_norm_single < 0) | (x_norm_single > 1)):
        st.warning(
            "Input berada di luar rentang data pelatihan; prediksi merupakan ekstrapolasi.",
            icon=":material/warning:",
        )

    mu, w, w_bar, f, o4, prediction = model.forward_layers(x_norm_single)
    if not all(np.all(np.isfinite(a)) for a in (mu, w, w_bar, f, o4, prediction)):
        st.error(
            "Perhitungan menghasilkan nilai non-finite. Periksa input dan latih ulang model."
        )
        return

    if w_bar.sum() < 1e-6:
        st.warning(
            "Aturan hampir tidak aktif untuk input ini; prediksi tidak dapat diandalkan.",
            icon=":material/warning:",
        )

    pred_minutes = float(denormalize_output(prediction[0], scaler_info))
    if y_actual_sample is not None:
        m1, m2 = st.columns(2)
        m1.metric(
            ":material/timer: Hasil Prediksi Waktu Pengantaran",
            f"{pred_minutes:.2f} menit",
            border=True,
        )
        m2.metric(
            ":material/fact_check: Waktu Aktual Data Uji",
            f"{y_actual_sample:.2f} menit",
            delta=f"Selisih: {pred_minutes - y_actual_sample:+.2f} menit",
            delta_color="off",
            border=True,
        )
    else:
        st.metric(
            ":material/timer: Hasil Prediksi Waktu Pengantaran",
            f"{pred_minutes:.2f} menit",
            border=True,
        )

    st.divider()
    render_layer_inspection(model, raw_vec, x_norm_single, scaler_info)

    # ==========================================================================
    # RINCIAN HITUNG MANUAL ESTAFET (LAPISAN 1 S.D. LAPISAN 5)
    # ==========================================================================
    st.divider()
    st.subheader(":material/calculate: Rincian Hitung Manual Estafet (Lapisan 1 – 5)")

    dominant_rule = int(np.argmax(w_bar[0]))

    def _format_rule_option(r_idx):
        tag_dom = " [Paling Dominan]" if r_idx == dominant_rule else ""
        return f"R{r_idx + 1} (Bobot w_bar = {w_bar[0, r_idx]:.4f}){tag_dom}"

    rule = st.selectbox(
        "Pilih Aturan yang Dijabarkan (Otomatis mengarah ke aturan paling dominan)",
        range(model.n_rules),
        index=dominant_rule,
        format_func=_format_rule_option,
        key="tab4_rule",
    )

    selected_mfs = model.rules[rule]
    mf_labels = ["Rendah (j=0)", "Tinggi (j=1)"]

    # --- LAPISAN 1: FUZZIFIKASI GAUSSIAN ---
    st.markdown("**Tahap 1 — Lapisan 1: Fuzzifikasi Gaussian**")
    st.latex(
        r"\mu_{i,j}(x_i) = \exp\left(-\frac{1}{2}\left(\frac{x_i - c_{i,j}}{\sigma_{i,j}}\right)^2\right)"
    )
    l1_lines = []
    selected_mu_vals = []
    for i, mf_idx in enumerate(selected_mfs):
        x_val = float(x_norm_single[0, i])
        c_val = float(model.c[i, mf_idx])
        s_val = float(model.sigma[i, mf_idx])
        mu_val = float(mu[0, i, mf_idx])
        selected_mu_vals.append(mu_val)
        l1_lines.append(
            f"x{i + 1} ({FEATURE_COLS[i]} -> {mf_labels[mf_idx]}): "
            f"exp(-0.5 * (({x_val:.6f} - {c_val:.6f}) / {s_val:.6f})^2) = {mu_val:.8f}"
        )
    st.code("\n".join(l1_lines), language="text")

    # --- LAPISAN 2: FIRING STRENGTH ---
    st.markdown(f"**Tahap 2 — Lapisan 2: Firing Strength Aturan R{rule + 1}**")
    st.latex(r"w_r = \prod_{i=1}^{5} \mu_{i, j_r}(x_i)")
    perkalian_str = " × ".join(f"{val:.8f}" for val in selected_mu_vals)
    st.code(f"w_R{rule + 1} = {perkalian_str} = {w[0, rule]:.10f}", language="text")

    # --- LAPISAN 3: NORMALISASI FIRING STRENGTH ---
    st.markdown(f"**Tahap 3 — Lapisan 3: Normalisasi Bobot Aturan R{rule + 1}**")
    st.latex(r"\bar{w}_r = \frac{w_r}{\sum_{k=1}^{32} w_k}")
    total_w = float(np.sum(w[0]))
    st.code(
        f"Total_w (sum R1..R32) = {total_w:.10f}\n"
        f"w_bar_R{rule + 1} = {w[0, rule]:.10f} / {total_w:.10f} "
        f"= {w_bar[0, rule]:.10f} ({w_bar[0, rule] * 100:.2f}%)",
        language="text",
    )

    # --- LAPISAN 4: DEFUZZIFIKASI SUGENO ORDE-1 ---
    st.markdown(f"**Tahap 4 — Lapisan 4: Fungsi Konsekuen Sugeno Orde-1 (R{rule + 1})**")
    st.latex(r"f_r = \sum_{i=1}^{5} p_{r,i} x_i + q_r, \qquad O_{4,r} = \bar{w}_r f_r")
    terms = model.consequents[rule, :5] * x_norm_single[0]
    detail = pd.DataFrame(
        {
            "Variabel": FEATURE_COLS + ["Bias"],
            "Input ternormalisasi": [*x_norm_single[0], 1.0],
            "Koefisien": model.consequents[rule],
            "Hasil perkalian": [*terms, model.consequents[rule, -1]],
        }
    )
    st.dataframe(detail, hide_index=True, use_container_width=True)
    st.code(
        f"f_R{rule + 1}  = "
        + " + ".join(f"({v:.10f})" for v in [*terms, model.consequents[rule, -1]])
        + f" = {f[0, rule]:.10f}\n"
        f"O4_R{rule + 1} = {w_bar[0, rule]:.10f} × {f[0, rule]:.10f} = {o4[0, rule]:.10f}",
        language="text",
    )

    # --- LAPISAN 5: AGREGASI AKHIR & DENORMALISASI ---
    st.markdown("**Tahap 5 — Lapisan 5: Agregasi Akhir & Denormalisasi ke Satuan Menit**")
    st.latex(
        r"\hat{y}_{\text{norm}} = \sum_{r=1}^{32} O_{4,r}, \qquad "
        r"\hat{y}_{\text{menit}} = \hat{y}_{\text{norm}} \cdot (y_{\max} - y_{\min}) + y_{\min}"
    )
    y_min = float(scaler_info["y_min"])
    y_max = float(scaler_info["y_max"])
    sisa_o4 = float(prediction[0] - o4[0, rule])
    st.code(
        f"y_pred_norm  = O4_R{rule + 1} + (jumlah 31 aturan lainnya)\n"
        f"             = {o4[0, rule]:.10f} + ({sisa_o4:.10f}) = {prediction[0]:.10f}\n"
        f"y_pred_menit = {prediction[0]:.10f} × ({y_max:.4f} - {y_min:.4f}) + {y_min:.4f} "
        f"= {pred_minutes:.4f} menit",
        language="text",
    )

    # Ekspor tanpa pembulatan tampilan, dengan input dan parameter per aturan.
    export = pd.DataFrame(
        {
            "Rule": [f"R{r + 1}" for r in range(model.n_rules)],
            "w": w[0],
            "w_bar": w_bar[0],
            "f": f[0],
            "o4": o4[0],
        }
    )
    for i in range(5):
        export[f"x{i + 1}_raw"] = raw_vec[i]
        export[f"x{i + 1}_norm"] = x_norm_single[0, i]
        export[f"mu{i + 1}_low"] = mu[0, i, 0]
        export[f"mu{i + 1}_high"] = mu[0, i, 1]
        export[f"mf{i + 1}"] = [r[i] for r in model.rules]
        export[f"p{i + 1}"] = model.consequents[:, i]
    export["q"] = model.consequents[:, -1]
    export["y_pred_norm"] = prediction[0]
    export["y_min"] = scaler_info["y_min"]
    export["y_max"] = scaler_info["y_max"]
    st.download_button(
        "Unduh perhitungan 32 aturan (CSV)",
        export.to_csv(index=False).encode("utf-8-sig"),
        "inspeksi_lapisan1_5.csv",
        "text/csv",
        icon=":material/download:",
        key="tab4_export",
    )
