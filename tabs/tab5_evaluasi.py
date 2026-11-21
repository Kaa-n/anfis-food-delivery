"""
Tab 5: Evaluasi Model
- Metrik KPI pada 20% data uji: RMSE, MAE, dan Koefisien Determinasi (R2).
- Grafik perbandingan Waktu Aktual vs Prediksi ANFIS (60 sampel uji pertama).
- Tabel perbandingan nilai aktual, prediksi, dan selisih error absolut.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
from tabs.components import denormalize_output


def render_tab5_evaluasi():
    st.subheader("Evaluasi Performa Model pada Data Uji (20%)")

    if st.session_state.model is None:
        st.warning(
            "Silakan latih model ANFIS terlebih dahulu pada Tab 1 atau Tab 3.",
            icon=":material/warning:",
        )
        return

    model = st.session_state.model
    _, _, X_test, _, y_test, scaler_info = st.session_state.data_pack

    _, _, _, _, _, y_pred_test_norm = model.forward_layers(X_test)
    y_actual_min = denormalize_output(y_test, scaler_info)
    y_pred_min = denormalize_output(y_pred_test_norm, scaler_info)

    rmse = float(np.sqrt(mean_squared_error(y_actual_min, y_pred_min)))
    mae = float(mean_absolute_error(y_actual_min, y_pred_min))
    r2 = float(r2_score(y_actual_min, y_pred_min))

    m1, m2, m3 = st.columns(3)
    m1.metric(
        ":material/square_foot: RMSE (Root Mean Squared Error)",
        f"{rmse:.2f} menit",
        border=True,
    )
    m2.metric(
        ":material/straighten: MAE (Mean Absolute Error)",
        f"{mae:.2f} menit",
        border=True,
    )
    m3.metric(
        ":material/monitoring: Koefisien Determinasi (R²)",
        f"{r2:.4f}",
        border=True,
    )

    limit = min(60, len(y_actual_min))
    st.markdown(
        f"**Perbandingan Waktu Aktual vs Prediksi ANFIS ({limit} Sampel Uji Pertama)**"
    )
    fig_eval, ax_eval = plt.subplots(figsize=(9.5, 3.5))
    ax_eval.plot(
        range(limit),
        y_actual_min[:limit],
        label="Waktu Aktual",
        color="#1F77B4",
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    ax_eval.plot(
        range(limit),
        y_pred_min[:limit],
        label="Prediksi ANFIS",
        color="#FF7F0E",
        linestyle="--",
        marker="x",
        markersize=4,
        linewidth=1.8,
    )
    ax_eval.set_xlabel("Indeks Sampel Uji")
    ax_eval.set_ylabel("Waktu Pengantaran (menit)")
    ax_eval.legend(loc="upper right")
    ax_eval.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
    ax_eval.spines["top"].set_visible(False)
    ax_eval.spines["right"].set_visible(False)
    st.pyplot(fig_eval)
    plt.close(fig_eval)

    df_comp = pd.DataFrame(
        {
            "Sampel #": np.arange(1, len(y_actual_min) + 1),
            "Waktu Aktual (menit)": np.round(y_actual_min, 2),
            "Prediksi ANFIS (menit)": np.round(y_pred_min, 2),
            "Selisih Error Absolut (menit)": np.round(np.abs(y_actual_min - y_pred_min), 2),
        }
    )
    st.dataframe(df_comp.head(20), use_container_width=True, hide_index=True)

    with st.expander("Analisis Kualitas & Keterbatasan Sistem", icon=":material/insights:"):
        st.markdown(
            """
            * **Kualitas Model:** Kombinasi *Least Squares Estimator* dan *Gradient Descent* mampu memetakan hubungan non-linear antar-variabel dengan error rata-rata (MAE) yang rendah dalam satuan menit.
            * **Keterbatasan:** Variabel kategorikal seperti cuaca dan kemacetan masih menggunakan penyederhanaan skala ordinal (`1–3` dan `1–5`), serta belum memperhitungkan faktor mendadak di lapangan seperti kendala teknis kendaraan kurir.
            """
        )