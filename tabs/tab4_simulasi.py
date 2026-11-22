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
        st.warning("Siapkan dataset di Tab 2 dan latih model di Tab 3.",
                   icon=":material/warning:")
        return
    model = st.session_state.model
    _, _, X_test, _, _, scaler_info = st.session_state.data_pack
    mode = st.radio("Sumber input", ["Input manual", "Sampel data uji"],
                    horizontal=True, key="tab4_source")
    if mode == "Sampel data uji":
        if len(X_test) == 0:
            st.warning("Data uji kosong.", icon=":material/warning:")
            return
        index = st.number_input("Nomor sampel data uji", min_value=1,
                                max_value=len(X_test), value=1, step=1,
                                key="tab4_sample")
        x_norm_single = np.asarray(X_test[int(index) - 1]).reshape(1, -1)
        raw_vec = (x_norm_single[0] *
                   (scaler_info["x_max"] - scaler_info["x_min"] + 1e-8) +
                   scaler_info["x_min"])
        st.caption("Sampel dan bobot model ini dapat dipakai untuk hitung manual estafet.")
    else:
        c1, c2, c3 = st.columns(3)
        in_dist = c1.number_input("Jarak Restoran - Pelanggan (km)", 0.1, 50.0,
                                 6.5, 0.5, key="sim_dist")
        in_prep = c2.number_input("Waktu Persiapan Makanan (menit)", 1.0, 90.0,
                                 15.0, 1.0, key="sim_prep")
        in_exp = c3.number_input("Pengalaman Kurir (tahun)", 0.0, 30.0,
                                3.0, 0.5, key="sim_exp")
        c4, c5 = st.columns(2)
        traffic = c4.selectbox("Tingkat Kemacetan Lalu Lintas",
                              list(TRAFFIC_MAP), index=1, key="sim_traf")
        weather = c5.selectbox("Kondisi Cuaca", list(WEATHER_MAP), key="sim_weat")
        raw_vec = [in_dist, in_prep, in_exp, TRAFFIC_MAP[traffic], WEATHER_MAP[weather]]
        x_norm_single = normalize_single_input(raw_vec, scaler_info).reshape(1, -1)

    if np.any((x_norm_single < 0) | (x_norm_single > 1)):
        st.warning("Input berada di luar rentang data pelatihan; prediksi merupakan ekstrapolasi.",
                   icon=":material/warning:")
    mu, w, w_bar, f, o4, prediction = model.forward_layers(x_norm_single)
    if not all(np.all(np.isfinite(a)) for a in (mu, w, w_bar, f, o4, prediction)):
        st.error("Perhitungan menghasilkan nilai non-finite. Periksa input dan latih ulang model.")
        return
    if w_bar.sum() < 1e-6:
        st.warning("Aturan hampir tidak aktif untuk input ini; prediksi tidak dapat diandalkan.",
                   icon=":material/warning:")
    st.metric(":material/timer: Hasil Prediksi Waktu Pengantaran",
              f"{float(denormalize_output(prediction[0], scaler_info)):.2f} menit",
              border=True)
    st.divider()
    render_layer_inspection(model, raw_vec, x_norm_single, scaler_info)

    st.markdown("**Rincian hitung manual Lapisan 4**")
    st.latex(r"f_r=\sum_{i=1}^{5}p_{r,i}x_i+q_r,\qquad O_{4,r}=\bar{w}_r f_r")
    rule = st.selectbox("Aturan yang dijabarkan", range(model.n_rules),
                        format_func=lambda r: f"R{r + 1}", key="tab4_rule")
    terms = model.consequents[rule, :5] * x_norm_single[0]
    detail = pd.DataFrame({
        "Variabel": FEATURE_COLS + ["Bias"],
        "Input ternormalisasi": [*x_norm_single[0], 1.0],
        "Koefisien": model.consequents[rule],
        "Hasil perkalian": [*terms, model.consequents[rule, -1]],
    })
    st.dataframe(detail, hide_index=True, use_container_width=True)
    st.code(
        f"f_R{rule + 1} = " +
        " + ".join(f"({v:.10f})" for v in [*terms, model.consequents[rule, -1]]) +
        f" = {f[0, rule]:.10f}\n"
        f"O4_R{rule + 1} = {w_bar[0, rule]:.10f} x {f[0, rule]:.10f}"
        f" = {o4[0, rule]:.10f}", language="text")
    # Ekspor tanpa pembulatan tampilan, dengan input dan parameter per aturan.
    export = pd.DataFrame({
        "Rule": [f"R{r + 1}" for r in range(model.n_rules)],
        "w": w[0], "w_bar": w_bar[0], "f": f[0], "o4": o4[0],
    })
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
    st.download_button("Unduh perhitungan 32 aturan (CSV)",
                       export.to_csv(index=False).encode("utf-8-sig"),
                       "inspeksi_lapisan4.csv", "text/csv",
                       icon=":material/download:", key="tab4_export")
