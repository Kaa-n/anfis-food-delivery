"""
Tab 2: Dataset & Preprocessing
- Informasi sumber dataset aktif (lokal CSV, unggahan pengguna, atau simulasi).
- Unggah file CSV Food_Delivery_Times.csv.
- Metrik ringkasan jumlah baris bersih, data latih (80%), dan data uji (20%).
- Pratinjau 10 baris data setelah cleaning & encoding serta statistik deskriptif.
"""

import pandas as pd
import streamlit as st
from tabs.components import FEATURE_COLS, TARGET_COL, preprocess_data


def render_tab2_dataset():
    st.subheader("Manajemen Dataset & Pra-pemrosesan")
    st.info(
        f"Sumber dataset aktif: **{st.session_state.data_source}**",
        icon=":material/info:",
    )

    uploaded = st.file_uploader(
        "Unggah Dataset CSV (Food_Delivery_Times.csv)",
        type=["csv"],
        key="tab2_uploader",
    )
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            data_pack_new = preprocess_data(df_up)
            st.session_state.df_raw = df_up
            st.session_state.data_pack = data_pack_new
            st.session_state.data_source = f"File Unggahan ({uploaded.name})"
            st.session_state.model = None
            st.success(
                "Dataset berhasil diunggah dan divalidasi! Silakan latih ulang model pada Tab 3.",
                icon=":material/check_circle:",
            )
        except Exception as e:
            st.error(f"Validasi gagal: {e}", icon=":material/error:")

    if st.session_state.data_pack is not None:
        df_clean, X_train, X_test, y_train, y_test, scaler_info = st.session_state.data_pack

        c1, c2, c3 = st.columns(3)
        c1.metric(
            ":material/table_rows: Total Data Bersih",
            f"{len(df_clean)} baris",
            border=True,
        )
        c2.metric(
            ":material/school: Data Latih (80%)",
            f"{len(X_train)} baris",
            border=True,
        )
        c3.metric(
            ":material/fact_check: Data Uji (20%)",
            f"{len(X_test)} baris",
            border=True,
        )

        st.markdown("**Pratinjau 10 Baris Data Setelah Cleaning & Encoding:**")
        st.dataframe(df_clean.head(10), use_container_width=True)

        with st.expander("Statistik Deskriptif & Keterangan Variabel", icon=":material/description:"):
            st.markdown(
                """
                * `Distance_km` ($x_1$): Jarak tempuh pengantaran dalam kilometer.
                * `Preparation_Time_min` ($x_2$): Waktu penyiapan pesanan di restoran dalam menit.
                * `Courier_Experience_yrs` ($x_3$): Lama pengalaman kerja kurir dalam tahun.
                * `Traffic_Level` ($x_4$): Tingkat kepadatan lalu lintas (`Low = 1`, `Medium = 2`, `High = 3`).
                * `Weather` ($x_5$): Kondisi cuaca (`Clear = 1`, `Windy = 2`, `Foggy = 3`, `Rainy = 4`, `Snowy = 5`).
                * `Delivery_Time_min` ($y$): Target waktu pengantaran aktual dalam menit.
                """
            )
            st.dataframe(df_clean.describe().T, use_container_width=True)
    else:
        st.error(
            f"Data belum tersedia atau tidak memiliki kolom wajib: {FEATURE_COLS + [TARGET_COL]}",
            icon=":material/error:",
        )