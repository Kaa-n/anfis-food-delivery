"""
Entry point utama aplikasi Streamlit:
Estimasi Waktu Tiba Pengantaran Makanan Berbasis ANFIS (5 Input, 2 MF Gaussian, 32 Aturan Sugeno).

Arsitektur modular:
- Logika algoritma & pemrosesan data berada di folder `src/`
- Tampilan antarmuka tiap tab dipisahkan ke dalam folder `tabs/`
"""

import os
import pandas as pd
import streamlit as st

from tabs import (
    render_tab1_beranda,
    render_tab2_dataset,
    render_tab3_training,
    render_tab4_simulasi,
    render_tab5_evaluasi,
    render_tab6_informasi,
)
from tabs.components import (
    STUB_MODULES,
    generate_fallback_dataset,
    inject_custom_css,
    preprocess_data,
)

st.set_page_config(
    page_title="Estimasi Waktu Tiba Pengantaran Makanan Berbasis ANFIS",
    page_icon=":material/delivery_dining:",
    layout="wide",
)

inject_custom_css()

# ==============================================================================
# INISIALISASI SESSION STATE & DATASET AWAL
# ==============================================================================
if "df_raw" not in st.session_state:
    default_path = os.path.join("data", "Food_Delivery_Times.csv")
    if os.path.exists(default_path):
        st.session_state.df_raw = pd.read_csv(default_path)
        st.session_state.data_source = "File Lokal (data/Food_Delivery_Times.csv)"
    else:
        st.session_state.df_raw = generate_fallback_dataset()
        st.session_state.data_source = "Dataset Simulasi Bawaan (Unggah CSV Kaggle di Tab 2)"

if "data_pack" not in st.session_state:
    try:
        st.session_state.data_pack = preprocess_data(st.session_state.df_raw)
    except Exception:
        st.session_state.data_pack = None

if "model" not in st.session_state:
    st.session_state.model = None

# ==============================================================================
# HEADER UTAMA & NAVIGASI 6 TAB HORIZONTAL
# ==============================================================================
st.title(":material/delivery_dining: Estimasi Waktu Tiba Pengantaran Makanan Berbasis ANFIS")
st.caption(
    "Adaptive Neuro-Fuzzy Inference System (5 Input, 2 MF Gaussian, 32 Aturan Sugeno Orde-1)"
)

if STUB_MODULES:
    st.caption(
        f":material/construction: Mode simulasi aktif untuk modul yang belum diisi: `{', '.join(STUB_MODULES)}`"
    )

tabs = st.tabs(
    [
        ":material/home: 1. Beranda & Prediksi Utama",
        ":material/database: 2. Dataset & Preprocessing",
        ":material/model_training: 3. Pelatihan Model (Training)",
        ":material/tune: 4. Simulasi & Inspeksi 5 Lapisan",
        ":material/analytics: 5. Evaluasi Model",
        ":material/menu_book: 6. Informasi Sistem & 32 Aturan",
    ]
)

with tabs[0]:
    render_tab1_beranda()

with tabs[1]:
    render_tab2_dataset()

with tabs[2]:
    render_tab3_training()

with tabs[3]:
    render_tab4_simulasi()

with tabs[4]:
    render_tab5_evaluasi()

with tabs[5]:
    render_tab6_informasi()