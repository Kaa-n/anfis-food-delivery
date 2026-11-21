"""
Tab 4: Simulasi & Inspeksi 5 Lapisan
- Formulir simulasi 5 parameter input secara bebas.
- Metrik hasil prediksi waktu pengantaran (menit).
- Tabel inspeksi lengkap Lapisan 1 (fuzzifikasi), Lapisan 2-4 (32 aturan Sugeno), dan Lapisan 5.
"""

import streamlit as st
from tabs.components import (
    TRAFFIC_MAP,
    WEATHER_MAP,
    denormalize_output,
    normalize_single_input,
    render_layer_inspection,
)


def render_tab4_simulasi():
    st.subheader("Simulasi Skenario & Validasi Perhitungan Per Lapisan")

    if st.session_state.model is None:
        st.warning(
            "Silakan latih model ANFIS terlebih dahulu pada Tab 1 atau Tab 3.",
            icon=":material/warning:",
        )
        return

    model = st.session_state.model
    _, _, _, _, _, scaler_info = st.session_state.data_pack

    c1, c2, c3 = st.columns(3)
    in_dist = c1.number_input(
        "Jarak Restoran - Pelanggan (km)",
        min_value=0.1,
        max_value=50.0,
        value=6.50,
        step=0.5,
        format="%.2f",
        key="sim_dist",
    )
    in_prep = c2.number_input(
        "Waktu Persiapan Makanan (menit)",
        min_value=1.0,
        max_value=90.0,
        value=15.0,
        step=1.0,
        key="sim_prep",
    )
    in_exp = c3.number_input(
        "Pengalaman Kurir (tahun)",
        min_value=0.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
        key="sim_exp",
    )

    c4, c5 = st.columns(2)
    in_traffic_str = c4.selectbox(
        "Tingkat Kemacetan Lalu Lintas",
        list(TRAFFIC_MAP.keys()),
        index=1,
        key="sim_traf",
    )
    in_weather_str = c5.selectbox(
        "Kondisi Cuaca",
        list(WEATHER_MAP.keys()),
        index=0,
        key="sim_weat",
    )

    raw_vec = [
        in_dist,
        in_prep,
        in_exp,
        TRAFFIC_MAP[in_traffic_str],
        WEATHER_MAP[in_weather_str],
    ]
    x_norm_single = normalize_single_input(raw_vec, scaler_info).reshape(1, -1)
    _, _, _, _, _, y_pred_norm = model.forward_layers(x_norm_single)
    pred_minutes = float(denormalize_output(y_pred_norm[0], scaler_info))

    col_met, _ = st.columns([1, 2])
    with col_met:
        st.metric(
            ":material/timer: Hasil Prediksi Waktu Pengantaran",
            f"{pred_minutes:.2f} menit",
            border=True,
        )

    st.divider()
    render_layer_inspection(model, raw_vec, x_norm_single, scaler_info)