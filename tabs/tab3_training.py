"""
Tab 3: Pelatihan Model (Training)
- Pengaturan parameter Hybrid Learning (Epoch, Learning Rate GD, Regularisasi LSE).
- Tombol pelatihan model ANFIS.
- Grafik Konvergensi Training (RMSE per Epoch).
- Grafik Adaptasi Fungsi Keanggotaan Gaussian (Sebelum vs Sesudah Training) untuk 5 fitur.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tabs.components import ANFISClass, FEATURE_COLS


def render_tab3_training():
    st.subheader("Konfigurasi & Pelatihan Hybrid Learning ANFIS")

    if st.session_state.data_pack is None:
        st.warning(
            "Pastikan dataset valid pada Tab 2 terlebih dahulu.",
            icon=":material/warning:",
        )
        return

    df_clean, X_train, X_test, y_train, y_test, scaler_info = st.session_state.data_pack

    col_p1, col_p2, col_p3 = st.columns(3)
    epochs = col_p1.slider(
        "Jumlah Epoch",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        key="tab3_epochs",
    )
    lr = col_p2.select_slider(
        "Learning Rate (Gradient Descent)",
        options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.05,
        key="tab3_lr",
    )
    reg = col_p3.select_slider(
        "Regularisasi LSE (Lambda)",
        options=[0.0001, 0.001, 0.01, 0.1],
        value=0.001,
        key="tab3_reg",
    )

    if st.button(
        "Latih Model ANFIS",
        type="primary",
        icon=":material/play_arrow:",
        key="tab3_train_btn",
    ):
        try:
            with st.spinner("Menjalankan Hybrid Learning (LSE + Gradient Descent)..."):
                anfis_model = ANFISClass(n_inputs=5, lr=lr, reg=reg)
                anfis_model.fit(X_train, y_train, epochs=epochs)
                if (not anfis_model.loss_history or
                        not np.all(np.isfinite(anfis_model.loss_history))):
                    raise ValueError("Riwayat RMSE kosong atau tidak finite.")
                st.session_state.model = anfis_model
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            st.error(f"Pelatihan gagal: {exc}", icon=":material/error:")

    if st.session_state.model is not None:
        model = st.session_state.model
        if not model.loss_history:
            st.info("Model belum memiliki riwayat pelatihan.")
            return
        final_prediction = model.forward_layers(X_train)[-1]
        final_rmse = float(np.sqrt(np.mean((final_prediction - y_train) ** 2)))
        st.metric("RMSE model tersimpan (normalisasi)", f"{final_rmse:.6f}")
        st.caption(
            f"Konfigurasi model: {len(model.loss_history)} epoch; "
            f"learning rate = {model.lr}; lambda = {model.reg}. "
            "Riwayat dicatat sesudah LSE, sebelum pembaruan premis GD tiap epoch."
        )
        history_df = pd.DataFrame({
            "Epoch": range(1, len(model.loss_history) + 1),
            "RMSE_normalisasi_sebelum_GD": model.loss_history,
        })
        st.download_button(
            "Unduh riwayat RMSE (CSV)",
            history_df.to_csv(index=False).encode("utf-8-sig"),
            "riwayat_training.csv", "text/csv",
            icon=":material/download:", key="tab3_export",
        )
        st.success(
            f"Pelatihan selesai! RMSE riwayat epoch terakhir (sebelum GD): {model.loss_history[-1]:.4f}",
            icon=":material/check_circle:",
        )

        st.markdown("**1. Grafik Konvergensi Training (RMSE per Epoch)**")
        fig_loss, ax_loss = plt.subplots(figsize=(9, 3.0))
        epochs_range = range(1, len(model.loss_history) + 1)
        ax_loss.plot(
            epochs_range,
            model.loss_history,
            color="#1F77B4",
            linewidth=2.2,
            label="RMSE Training",
        )
        ax_loss.scatter(
            [len(model.loss_history)],
            [model.loss_history[-1]],
            color="#FF4B4B",
            zorder=5,
            label=f"Akhir ({model.loss_history[-1]:.4f})",
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("RMSE (Normalisasi)")
        ax_loss.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
        ax_loss.spines["top"].set_visible(False)
        ax_loss.spines["right"].set_visible(False)
        ax_loss.legend(loc="upper right", fontsize=8)
        st.pyplot(fig_loss)
        plt.close(fig_loss)

        st.markdown(
            "**2. Adaptasi Fungsi Keanggotaan Gaussian (Sebelum vs Sesudah Training)**"
        )
        x_grid = np.linspace(0, 1, 200)
        fig_mf, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
        for i in range(5):
            ax = axes[i]
            ax.plot(
                x_grid,
                model.gaussian_mf(x_grid, model.init_c[i, 0], model.init_sigma[i, 0]),
                linestyle="--",
                color="#808495",
                alpha=0.7,
                linewidth=1.2,
                label="Awal (Initial)",
            )
            ax.plot(
                x_grid,
                model.gaussian_mf(x_grid, model.init_c[i, 1], model.init_sigma[i, 1]),
                linestyle="--",
                color="#808495",
                alpha=0.7,
                linewidth=1.2,
            )
            ax.plot(
                x_grid,
                model.gaussian_mf(x_grid, model.c[i, 0], model.sigma[i, 0]),
                color="#1F77B4",
                linewidth=2,
                label="Rendah (Trained)",
            )
            ax.plot(
                x_grid,
                model.gaussian_mf(x_grid, model.c[i, 1], model.sigma[i, 1]),
                color="#FF4B4B",
                linewidth=2,
                label="Tinggi (Trained)",
            )
            ax.set_title(FEATURE_COLS[i], fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Derajat Keanggotaan (μ)")
        axes[-1].legend(loc="upper right", fontsize=7)
        st.pyplot(fig_mf)
        plt.close(fig_mf)
