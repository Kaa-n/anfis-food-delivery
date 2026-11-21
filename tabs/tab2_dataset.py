"""
Tab 2: Dataset & Preprocessing
- Informasi sumber dataset aktif (lokal CSV, unggahan pengguna, atau simulasi).
- Unggah file CSV Food_Delivery_Times.csv.
- Metrik ringkasan jumlah baris bersih, data latih (80%), dan data uji (20%).
- Pratinjau 10 baris data setelah cleaning & encoding serta statistik deskriptif.
"""

import numpy as np
import pandas as pd
import streamlit as st

from tabs.components import FEATURE_COLS, TARGET_COL, preprocess_data


TRAFFIC_LABELS = {
    1.0: "Low",
    2.0: "Medium",
    3.0: "High",
}

WEATHER_LABELS = {
    1.0: "Clear",
    2.0: "Windy",
    3.0: "Foggy",
    4.0: "Rainy",
    5.0: "Snowy",
}


def create_histogram(series, bins=10):
    """Membuat data histogram untuk visualisasi Streamlit."""
    counts, edges = np.histogram(series.dropna(), bins=bins)

    labels = [
        f"{edges[i]:.1f} - {edges[i + 1]:.1f}"
        for i in range(len(counts))
    ]

    return pd.DataFrame(
        {"Jumlah Data": counts},
        index=labels
    )


def render_tab2_dataset():
    st.subheader("Manajemen Dataset & Pra-pemrosesan")

    # ==================================================
    # 1. INFORMASI SUMBER DATASET
    # ==================================================
    st.info(
        f"Sumber dataset aktif: **{st.session_state.data_source}**",
        icon=":material/info:",
    )

    # ==================================================
    # 2. UPLOAD DATASET
    # ==================================================
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
            st.session_state.data_source = (
                f"File Unggahan ({uploaded.name})"
            )
            st.session_state.model = None

            st.success(
                "Dataset berhasil diunggah dan divalidasi! "
                "Silakan latih ulang model pada Tab 3.",
                icon=":material/check_circle:",
            )

        except Exception as e:
            st.error(
                f"Validasi gagal: {e}",
                icon=":material/error:",
            )

    # ==================================================
    # 3. RINGKASAN DATASET
    # ==================================================
    if st.session_state.data_pack is None:
        st.error(
            f"Data belum tersedia atau tidak memiliki kolom wajib: "
            f"{FEATURE_COLS + [TARGET_COL]}",
            icon=":material/error:",
        )
        return

    (
        df_clean,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_info,
    ) = st.session_state.data_pack

    df_raw = st.session_state.df_raw

    total_raw = len(df_raw)
    total_clean = len(df_clean)
    total_removed = total_raw - total_clean

    st.markdown("### Ringkasan Dataset")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Data Awal",
        f"{total_raw:,} baris",
    )

    c2.metric(
        "Data Bersih",
        f"{total_clean:,} baris",
    )

    c3.metric(
        "Data Terhapus",
        f"{total_removed:,} baris",
    )

    c4.metric(
        "Jumlah Fitur",
        f"{len(FEATURE_COLS)} fitur",
    )

    st.caption(
        f"Data bersih: {len(X_train):,} data latih "
        f"dan {len(X_test):,} data uji."
    )

    # ==================================================
    # 4. PREVIEW DATA HASIL CLEANING
    # ==================================================
    st.markdown("### Hasil Data Cleaning & Encoding")

    st.write(
        "Berikut merupakan 10 baris pertama dataset "
        "setelah proses pembersihan dan encoding."
    )

    st.dataframe(
        df_clean.head(10),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Lihat Statistik Deskriptif",
        icon=":material/description:",
    ):
        st.write(
            "Statistik deskriptif menggambarkan karakteristik "
            "masing-masing variabel setelah preprocessing."
        )

        st.dataframe(
            df_clean.describe().T,
            use_container_width=True,
        )

        st.markdown("**Keterangan Variabel**")

        descriptions = {
            "Distance_km": "Jarak tempuh pengantaran (km).",
            "Preparation_Time_min": "Waktu penyiapan pesanan (menit).",
            "Courier_Experience_yrs": "Pengalaman kurir (tahun).",
            "Traffic_Level": "Tingkat lalu lintas (Low, Medium, High).",
            "Weather": "Kondisi cuaca.",
            "Delivery_Time_min": "Waktu pengantaran aktual (menit).",
        }

        for col in FEATURE_COLS + [TARGET_COL]:
            st.markdown(f"- **{col}**: {descriptions[col]}")

    # ==================================================
    # 5. VISUALISASI DISTRIBUSI DATA
    # ==================================================
    st.markdown("---")
    st.markdown("### Visualisasi Distribusi Data")

    st.write(
        "Visualisasi berikut menunjukkan sebaran nilai "
        "variabel numerik dan kategori pada dataset."
    )

    # Distribusi variabel numerik
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribusi Jarak Pengantaran**")
        st.bar_chart(
            create_histogram(df_clean["Distance_km"]),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Distribusi Waktu Persiapan**")
        st.bar_chart(
            create_histogram(df_clean["Preparation_Time_min"]),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Distribusi Pengalaman Kurir**")
        st.bar_chart(
            create_histogram(df_clean["Courier_Experience_yrs"]),
            use_container_width=True,
        )

    with col4:
        st.markdown("**Distribusi Waktu Pengantaran**")
        st.bar_chart(
            create_histogram(df_clean[TARGET_COL]),
            use_container_width=True,
        )

    # Distribusi kategori setelah ordinal encoding
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Distribusi Tingkat Lalu Lintas**")

        traffic_counts = (
            df_clean["Traffic_Level"]
            .value_counts()
            .sort_index()
        )

        traffic_counts.index = [
            TRAFFIC_LABELS.get(float(value), str(value))
            for value in traffic_counts.index
        ]

        st.bar_chart(
            traffic_counts,
            use_container_width=True,
        )

    with col6:
        st.markdown("**Distribusi Kondisi Cuaca**")

        weather_counts = (
            df_clean["Weather"]
            .value_counts()
            .sort_index()
        )

        weather_counts.index = [
            WEATHER_LABELS.get(float(value), str(value))
            for value in weather_counts.index
        ]

        st.bar_chart(
            weather_counts,
            use_container_width=True,
        )

    # ==================================================
    # 6. INFORMASI NORMALISASI MIN-MAX
    # ==================================================
    st.markdown("---")
    st.markdown("### Normalisasi Data (Min-Max Scaling)")

    st.write(
        "Normalisasi Min-Max digunakan untuk menyamakan skala "
        "variabel input dan target agar berada pada rentang "
        "mendekati 0 sampai 1."
    )

    st.latex(
        r"X_{norm} = \frac{X-X_{min}}"
        r"{X_{max}-X_{min}+\varepsilon}"
    )

    # Tabel rentang nilai awal
    scaling_rows = []

    for i, feature in enumerate(FEATURE_COLS):
        scaling_rows.append({
            "Variabel": feature,
            "Nilai Minimum": scaler_info["x_min"][i],
            "Nilai Maksimum": scaler_info["x_max"][i],
        })

    scaling_rows.append({
        "Variabel": TARGET_COL,
        "Nilai Minimum": scaler_info["y_min"],
        "Nilai Maksimum": scaler_info["y_max"],
    })

    st.markdown("**Rentang Nilai Sebelum Normalisasi**")

    st.dataframe(
        pd.DataFrame(scaling_rows).round(4),
        use_container_width=True,
        hide_index=True,
    )

    # ==================================================
    # 7. PREVIEW HASIL NORMALISASI
    # ==================================================
    st.markdown("**Preview Data Setelah Normalisasi**")

    normalized_df = pd.DataFrame(
        X_train[:10],
        columns=FEATURE_COLS,
    )

    normalized_df[TARGET_COL] = y_train[:10]

    st.dataframe(
        normalized_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Preview menampilkan 10 data pertama dari data latih "
        "yang telah dinormalisasi."
    )

    st.success(
        "Tahap cleaning, encoding, pembagian data, "
        "dan normalisasi berhasil dilakukan.",
        icon=":material/check_circle:",
    )
