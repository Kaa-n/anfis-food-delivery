"""
Tab 6: Informasi Sistem & 32 Aturan
- Panduan penggunaan 5 fitur utama aplikasi.
- Tabel parameter premis Gaussian terlatih (c0, sigma0, c1, sigma1).
- Matriks parameter konsekuen Sugeno Orde-1 (p1...p5, q) untuk seluruh 32 aturan.
"""

import numpy as np
import pandas as pd
import streamlit as st
from tabs.components import FEATURE_COLS


def render_tab6_informasi():
    st.subheader("Informasi Sistem, Parameter Akhir & Panduan Penggunaan")

    st.info(
        """
        **Panduan Penggunaan Aplikasi:**
        1. **Beranda & Prediksi Utama:** Memprediksi waktu pengantaran menggunakan kondisi operasional deterministik berbasis *seed* atau input jarak manual.
        2. **Dataset & Preprocessing:** Mengunggah atau memuat dataset CSV lokal, membersihkan nilai hilang, dan membagi data latih (80%) serta uji (20%).
        3. **Pelatihan Model (Training):** Mengatur *epoch*, *learning rate*, dan regularisasi LSE untuk melatih parameter premis dan konsekuen ANFIS.
        4. **Simulasi & Inspeksi 5 Lapisan:** Menguji skenario bebas dan memvalidasi derajat keanggotaan Gaussian hingga agregasi 32 aturan Sugeno.
        5. **Evaluasi Model:** Mengukur performa generalisasi model pada data uji menggunakan metrik RMSE, MAE, $R^2$, serta grafik aktual vs prediksi.
        """,
        icon=":material/help_outline:",
    )

    if st.session_state.model is not None:
        model = st.session_state.model
        st.markdown(r"**1. Parameter Premis Gaussian Terlatih ($c$ dan $\sigma$):**")
        df_prem = pd.DataFrame(
            {
                "Variabel Input": FEATURE_COLS,
                "Center Rendah (c0)": np.round(model.c[:, 0], 4),
                "Sigma Rendah (σ0)": np.round(model.sigma[:, 0], 4),
                "Center Tinggi (c1)": np.round(model.c[:, 1], 4),
                "Sigma Tinggi (σ1)": np.round(model.sigma[:, 1], 4),
            }
        )
        st.dataframe(df_prem, use_container_width=True, hide_index=True)

        st.markdown(
            r"**2. Matriks Parameter Konsekuen Sugeno Orde-1 ($p_1 \dots p_5, q$) untuk 32 Aturan:**"
        )
        df_cons = pd.DataFrame(
            np.round(model.consequents, 4),
            columns=[
                "p1 (Dist)",
                "p2 (Prep)",
                "p3 (Exp)",
                "p4 (Traffic)",
                "p5 (Weather)",
                "q (Konstanta)",
            ],
        )

        # tampilkan label kondisi premis jika atribut model.rules tersedia
        if hasattr(model, "rules") and model.rules is not None:
            premis_labels = [
                ", ".join(
                    f"{col.split('_')[0]}={'Rendah' if val == 0 else 'Tinggi'}"
                    for col, val in zip(FEATURE_COLS, rule)
                )
                for rule in model.rules
            ]
            df_cons.insert(0, "Kondisi Premis (IF)", premis_labels)


        df_cons.insert(0, "Aturan", [f"Rule {i + 1}" for i in range(model.n_rules)])
        st.dataframe(df_cons, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Latih model terlebih dahulu pada Tab 1 atau Tab 3 untuk menampilkan tabel parameter premis dan 32 aturan konsekuen.",
            icon=":material/info:",
        )