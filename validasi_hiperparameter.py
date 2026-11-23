"""
Skrip Validasi Silang (5-Fold Cross-Validation) untuk Seleksi Hiperparameter ANFIS.
Dijalankan terpisah dari GUI agar pemilihan hiperparameter murni menggunakan data latih (80%),
sementara data uji (20%) hanya dievaluasi satu kali di akhir untuk mencegah data leakage.

Cara menjalankan:
    python validasi_hiperparameter.py
"""

import itertools
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.anfis import ANFIS
from src.preprocessing import (
    denormalize_output,
    generate_fallback_dataset,
    preprocess_data,
)


def main():
    # 1. Muat & bersihkan dataset
    data_path = os.path.join("data", "Food_Delivery_Times.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        sumber = data_path
    else:
        df_raw = generate_fallback_dataset()
        sumber = "Dataset Simulasi Bawaan"

    df_clean, X_train, X_test, y_train, y_test, scaler_info = preprocess_data(
        df_raw, test_size=0.2, random_state=42
    )

    print("=" * 72)
    print("VALIDASI HIPERPARAMETER ANFIS (5-FOLD CROSS-VALIDATION PADA DATA LATIH)")
    print("=" * 72)
    print(f"Sumber Dataset : {sumber}")
    print(f"Total Bersih   : {len(df_clean)} baris")
    print(f"Data Latih     : {len(X_train)} baris (digunakan untuk 5-Fold CV)")
    print(f"Data Uji       : {len(X_test)} baris (disegel, hanya dipakai 1x di akhir)")
    print("-" * 72)

    # 2. Definisikan Grid Hiperparameter Terkendali (3 x 4 x 4 = 48 kombinasi)
    grid_epochs = [10, 50, 200]
    grid_lr = [0.001, 0.01, 0.05, 0.1]
    grid_reg = [0.0001, 0.001, 0.01, 0.1]

    kombinasi = list(itertools.product(grid_epochs, grid_lr, grid_reg))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    hasil_grid = []
    mulai = time.time()

    print(f"Menguji {len(kombinasi)} kombinasi hiperparameter (total {len(kombinasi) * 5} pelatihan)...")

    for idx, (epochs, lr, reg) in enumerate(kombinasi, start=1):
        fold_rmse_norm = []
        fold_rmse_menit = []
        fold_mae_menit = []
        fold_r2 = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = ANFIS(n_inputs=5, lr=lr, reg=reg)
            model.fit(X_tr_fold, y_tr_fold, epochs=epochs)

            _, _, _, _, _, y_val_pred_norm = model.forward_layers(X_val_fold)

            # Hitung error validasi (skala normalisasi & satuan menit)
            rmse_n = float(np.sqrt(mean_squared_error(y_val_fold, y_val_pred_norm)))
            y_val_asli = denormalize_output(y_val_fold, scaler_info)
            y_val_pred = denormalize_output(y_val_pred_norm, scaler_info)

            rmse_m = float(np.sqrt(mean_squared_error(y_val_asli, y_val_pred)))
            mae_m = float(mean_absolute_error(y_val_asli, y_val_pred))
            r2_val = float(r2_score(y_val_asli, y_val_pred))

            fold_rmse_norm.append(rmse_n)
            fold_rmse_menit.append(rmse_m)
            fold_mae_menit.append(mae_m)
            fold_r2.append(r2_val)

        hasil_grid.append(
            {
                "Epoch": epochs,
                "Learning_Rate": lr,
                "Lambda_LSE": reg,
                "RMSE_Validasi_Norm": round(float(np.mean(fold_rmse_norm)), 6),
                "RMSE_Validasi_Menit": round(float(np.mean(fold_rmse_menit)), 4),
                "Std_RMSE_Menit": round(float(np.std(fold_rmse_menit)), 4),
                "MAE_Validasi_Menit": round(float(np.mean(fold_mae_menit)), 4),
                "R2_Validasi": round(float(np.mean(fold_r2)), 4),
            }
        )

        if idx % 12 == 0 or idx == len(kombinasi):
            print(f"  Progres: {idx}/{len(kombinasi)} kombinasi selesai...")

    durasi = time.time() - mulai

    # 3. Urutkan berdasarkan RMSE Validasi terendah & tandai pemenang
    df_hasil = pd.DataFrame(hasil_grid).sort_values(
        by="RMSE_Validasi_Menit", ascending=True
    ).reset_index(drop=True)

    df_hasil.insert(0, "Peringkat", np.arange(1, len(df_hasil) + 1))
    df_hasil["Status"] = ["Dipilih" if i == 0 else "Tidak" for i in range(len(df_hasil))]

    output_csv = "hasil_validasi_kfold.csv"
    df_hasil.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("-" * 72)
    print(f"Selesai dalam {durasi:.2f} detik. Seluruh hasil disimpan ke '{output_csv}'.")
    print("\nTOP 10 KOMBINASI BERDASARKAN RMSE VALIDASI TERENDAH:")
    print(df_hasil.head(10).to_string(index=False))

    # 4. Evaluasi Akhir SATU KALI pada Data Uji (20% yang disegel)
    terbaik = df_hasil.iloc[0]
    best_epoch = int(terbaik["Epoch"])
    best_lr = float(terbaik["Learning_Rate"])
    best_reg = float(terbaik["Lambda_LSE"])

    print("\n" + "=" * 72)
    print("EVALUASI AKHIR MODEL TERPILIH PADA DATA UJI (20% / 182 BARIS)")
    print("=" * 72)
    print(
        f"Konfigurasi Terpilih : Epoch = {best_epoch}, "
        f"Learning Rate = {best_lr}, Lambda = {best_reg}"
    )

    final_model = ANFIS(n_inputs=5, lr=best_lr, reg=best_reg)
    final_model.fit(X_train, y_train, epochs=best_epoch)

    _, _, _, _, _, y_test_pred_norm = final_model.forward_layers(X_test)
    y_test_asli = denormalize_output(y_test, scaler_info)
    y_test_pred = denormalize_output(y_test_pred_norm, scaler_info)

    test_rmse = float(np.sqrt(mean_squared_error(y_test_asli, y_test_pred)))
    test_mae = float(mean_absolute_error(y_test_asli, y_test_pred))
    test_r2 = float(r2_score(y_test_asli, y_test_pred))

    print(f"RMSE Data Uji        : {test_rmse:.2f} menit")
    print(f"MAE Data Uji         : {test_mae:.2f} menit")
    print(f"R² Data Uji          : {test_r2:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()