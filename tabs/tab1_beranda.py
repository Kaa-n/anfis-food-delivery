"""
Tab 1: Beranda & Prediksi Utama
- Deskripsi & tujuan sistem ANFIS.
- Kartu kondisi operasional terkini berbasis seed tanggal & jam (dengan mode simulasi waktu demo).
- Formulir prediksi cepat (mode otomatis seed vs input manual penuh).
- Metrik hasil prediksi + status klasifikasi waktu pengantaran + expander inspeksi 5 lapisan.
"""

from datetime import datetime
import streamlit as st
from tabs.components import (
    ANFISClass,
    TRAFFIC_MAP,
    WEATHER_MAP,
    denormalize_output,
    get_current_operational_context,
    normalize_single_input,
    render_layer_inspection,
)


def render_tab1_beranda():
    st.subheader("Deskripsi & Tujuan Sistem")
    st.write(
        "Sistem ini mengestimasi waktu tiba pengantaran makanan secara adaptif menggunakan "
        "arsitektur **Adaptive Neuro-Fuzzy Inference System (ANFIS)**. Model memetakan interaksi "
        "non-linear antara lima variabel operasional utama—jarak tempuh restoran ke pelanggan, "
        "waktu persiapan dapur, pengalaman kurir, tingkat kemacetan lalu lintas, dan kondisi "
        "cuaca—ke dalam 32 aturan fuzzy Sugeno Orde-1 dengan fungsi keanggotaan Gaussian yang "
        "dilatih secara hibrida."
    )

    st.divider()

    # Bagian Kondisi Operasional Berdasarkan Seed Tanggal & Jam
    st.subheader(":material/sensors: Kondisi Operasional Saat Ini (Seed Tanggal & Jam)")
    use_custom_time = st.checkbox(
        "Simulasikan tanggal/jam tertentu (untuk menguji perubahan seed saat demo)",
        key="home_use_custom_time",
    )
    if use_custom_time:
        sc1, sc2 = st.columns(2)
        sim_date = sc1.date_input("Pilih Tanggal Simulasi", value=datetime.now(), key="home_sim_date")
        sim_hour = sc2.slider(
            "Pilih Jam Simulasi (00–23)",
            min_value=0,
            max_value=23,
            value=datetime.now().hour,
            key="home_sim_hour",
        )
        custom_dt = datetime(sim_date.year, sim_date.month, sim_date.day, sim_hour)
        ctx = get_current_operational_context(custom_dt)
    else:
        ctx = get_current_operational_context()

    st.caption(
        f"Waktu Referensi: **{ctx['timestamp_str']}** | Seed Deterministik: `{ctx['seed']}`"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(":material/partly_cloudy_day: Cuaca Saat Ini", ctx["Weather"], border=True)
    k2.metric(":material/traffic: Tingkat Kemacetan", ctx["Traffic_Level"], border=True)
    k3.metric(
        ":material/skillet: Estimasi Antrean Dapur",
        f"{ctx['Preparation_Time_min']:.0f} menit",
        border=True,
    )
    k4.metric(
        ":material/badge: Pengalaman Kurir",
        f"{ctx['Courier_Experience_yrs']} tahun",
        border=True,
    )

    st.divider()

    # Formulir Prediksi Cepat di Halaman Utama
    st.subheader(":material/bolt: Prediksi Waktu Pengantaran")
    mode_input = st.radio(
        "Pilih Sumber Parameter Input:",
        [
            "Gunakan Kondisi Saat Ini (Otomatis dari Seed)",
            "Input Manual Penuh (Simulasi Bebas)",
        ],
        horizontal=True,
        key="home_mode_input",
    )

    if mode_input == "Gunakan Kondisi Saat Ini (Otomatis dari Seed)":
        col_dist, _ = st.columns([1, 2])
        with col_dist:
            in_dist_home = st.number_input(
                "Masukkan Jarak Restoran ke Pelanggan (km)",
                min_value=0.1,
                max_value=50.0,
                value=5.50,
                step=0.5,
                format="%.2f",
                key="home_dist_auto",
            )
        in_prep_home = float(ctx["Preparation_Time_min"])
        in_exp_home = float(ctx["Courier_Experience_yrs"])
        in_traffic_home = ctx["Traffic_Level"]
        in_weather_home = ctx["Weather"]
    else:
        hm1, hm2, hm3 = st.columns(3)
        in_dist_home = hm1.number_input(
            "Jarak Pengantaran (km)",
            min_value=0.1,
            max_value=50.0,
            value=5.50,
            step=0.5,
            format="%.2f",
            key="home_dist_man",
        )
        in_prep_home = hm2.number_input(
            "Waktu Persiapan Dapur (menit)",
            min_value=1.0,
            max_value=90.0,
            value=15.0,
            step=1.0,
            key="home_prep_man",
        )
        in_exp_home = hm3.number_input(
            "Pengalaman Kurir (tahun)",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
            key="home_exp_man",
        )
        hm4, hm5 = st.columns(2)
        in_traffic_home = hm4.selectbox(
            "Tingkat Kemacetan",
            list(TRAFFIC_MAP.keys()),
            index=1,
            key="home_traf_man",
        )
        in_weather_home = hm5.selectbox(
            "Kondisi Cuaca",
            list(WEATHER_MAP.keys()),
            index=0,
            key="home_weat_man",
        )

    if st.session_state.model is None:
        st.warning(
            "Model ANFIS belum dilatih. Anda dapat melatihnya secara lengkap di **Tab 3**, "
            "atau tekan tombol cepat di bawah ini.",
            icon=":material/warning:",
        )
        if st.button(
            "Latih Model Default Sekarang (50 Epoch)",
            type="primary",
            icon=":material/rocket_launch:",
            key="home_quick_train",
        ):
            if st.session_state.data_pack is not None:
                _, X_tr, _, y_tr, _, _ = st.session_state.data_pack
                quick_model = ANFISClass(n_inputs=5, lr=0.05, reg=1e-3)
                quick_model.fit(X_tr, y_tr, epochs=50)
                st.session_state.model = quick_model
                st.rerun()
            else:
                st.error(
                    "Dataset belum valid. Periksa Tab 2 terlebih dahulu.",
                    icon=":material/error:",
                )
    else:
        _, _, _, _, _, scaler_info = st.session_state.data_pack
        raw_home = [
            in_dist_home,
            in_prep_home,
            in_exp_home,
            TRAFFIC_MAP[in_traffic_home],
            WEATHER_MAP[in_weather_home],
        ]
        x_norm_home = normalize_single_input(raw_home, scaler_info).reshape(1, -1)
        _, _, _, _, _, y_pred_home_norm = st.session_state.model.forward_layers(x_norm_home)
        pred_home_min = float(denormalize_output(y_pred_home_norm[0], scaler_info))

        res1, res2 = st.columns([1, 2])
        with res1:
            st.metric(
                ":material/timer: Estimasi Waktu Tiba",
                f"{pred_home_min:.1f} menit",
                border=True,
            )
        with res2:
            if pred_home_min <= 30:
                st.success(
                    "**Status: Pengantaran Cepat** — Hambatan jalan dan waktu dapur tergolong rendah.",
                    icon=":material/check_circle:",
                )
            elif pred_home_min <= 55:
                st.info(
                    "**Status: Pengantaran Normal** — Waktu pengiriman berada pada rentang standar.",
                    icon=":material/info:",
                )
            else:
                st.warning(
                    "**Status: Potensi Keterlambatan Tinggi** — Jarak jauh, antrean dapur, atau cuaca menghambat pengiriman.",
                    icon=":material/warning:",
                )

        with st.expander("Lihat Rincian Komputasi 5 Lapisan ANFIS", icon=":material/search:"):
            render_layer_inspection(st.session_state.model, raw_home, x_norm_home, scaler_info)

    with st.expander("Ringkasan Arsitektur 5 Lapisan ANFIS", icon=":material/account_tree:"):
        st.markdown(
            r"""
            * **Lapisan 1 (Fuzzifikasi):** 10 node Gaussian MF (2 per input: $\mu_{i,0}$ *Rendah*, $\mu_{i,1}$ *Tinggi*).
            * **Lapisan 2 (Firing Strength):** 32 node produk aturan ($w_r = \prod_{i=1}^{5} \mu_{i, j_r}$).
            * **Lapisan 3 (Normalisasi):** Menghitung bobot relatif tiap aturan ($\bar{w}_r = w_r / \sum_{k=1}^{32} w_k$).
            * **Lapisan 4 (Defuzzifikasi Sugeno Orde-1):** 32 fungsi linear orde-1 ($\bar{w}_r f_r = \bar{w}_r (p_{r,1}x_1 + \dots + p_{r,5}x_5 + q_r)$).
            * **Lapisan 5 (Output Akhir):** Penjumlahan terbobot tunggal ($\hat{y} = \sum_{r=1}^{32} \bar{w}_r f_r$) lalu didenormalisasi ke satuan menit.
            """
        )