"""
Skrip Ekstraksi Model Terbaik ANFIS.
Melatih model sekali dengan konfigurasi optimal hasil 5-Fold CV
(Epoch=10, LR=0.001, Lambda=0.1) dan menyimpannya ke folder models/.

Cara menjalankan:
    python ekstrak_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.anfis import ANFIS
from src.preprocessing import (
    denormalize_output,
    generate_fallback_dataset,
    preprocess_data,
)

MODEL_PATH = os.path.join("models", "model_anfis_terbaik.pkl")


def ekstrak_dan_simpan(output_path=MODEL_PATH):
    # 1. Muat & pra-proses dataset
    data_path = os.path.join("data", "Food_Delivery_Times.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = generate_fallback_dataset()

    df_clean, X_train, X_test, y_train, y_test, scaler_info = preprocess_data(
        df_raw, test_size=0.2, random_state=42
    )

    # 2. Latih model dengan konfigurasi optimal terpilih
    best_epochs = 10
    best_lr = 0.001
    best_reg = 0.1

    model = ANFIS(n_inputs=5, lr=best_lr, reg=best_reg)
    model.fit(X_train, y_train, epochs=best_epochs)

    # 3. Verifikasi metrik akhir sebelum disimpan
    _, _, _, _, _, y_train_pred = model.forward_layers(X_train)
    rmse_train_norm = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))

    _, _, _, _, _, y_test_pred_norm = model.forward_layers(X_test)
    y_test_asli = denormalize_output(y_test, scaler_info)
    y_test_pred = denormalize_output(y_test_pred_norm, scaler_info)

    rmse_test = float(np.sqrt(mean_squared_error(y_test_asli, y_test_pred)))
    mae_test = float(mean_absolute_error(y_test_asli, y_test_pred))
    r2_test = float(r2_score(y_test_asli, y_test_pred))

    # 4. Kemas seluruh parameter model & scaler ke dalam satu artefak
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    artefak = {
        "n_inputs": model.n_inputs,
        "n_mfs": model.n_mfs,
        "lr": model.lr,
        "reg": model.reg,
        "epochs": best_epochs,
        "rules": model.rules,
        "c": model.c,
        "sigma": model.sigma,
        "init_c": model.init_c,
        "init_sigma": model.init_sigma,
        "consequents": model.consequents,
        "loss_history": model.loss_history,
        "scaler_info": scaler_info,
        "split_config": {"test_size": 0.2, "random_state": 42},
        "metrics": {
            "rmse_train_norm": rmse_train_norm,
            "rmse_test_menit": rmse_test,
            "mae_test_menit": mae_test,
            "r2_test": r2_test,
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(artefak, f)

    print("=" * 65)
    print("EKSTRAKSI MODEL ANFIS BERHASIL")
    print("=" * 65)
    print(f"File Tersimpan          : {output_path}")
    print(f"Konfigurasi             : Epoch={best_epochs}, LR={best_lr}, Lambda={best_reg}")
    print(f"RMSE Latih (Normalisasi): {rmse_train_norm:.6f}")
    print(f"RMSE Uji                : {rmse_test:.2f} menit")
    print(f"MAE Uji                 : {mae_test:.2f} menit")
    print(f"R² Uji                  : {r2_test:.4f}")
    print("=" * 65)


def muat_model_terbaik(model_path=MODEL_PATH):
    """Memuat artefak model tersimpan dan mengembalikan objek ANFIS siap pakai."""
    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        artefak = pickle.load(f)

    model = ANFIS(
        n_inputs=artefak["n_inputs"],
        lr=artefak["lr"],
        reg=artefak["reg"],
    )
    model.rules = artefak["rules"]
    model.n_rules = len(artefak["rules"])
    model.c = artefak["c"]
    model.sigma = artefak["sigma"]
    model.init_c = artefak["init_c"]
    model.init_sigma = artefak["init_sigma"]
    model.consequents = artefak["consequents"]
    model.loss_history = artefak["loss_history"]

    return model, artefak["scaler_info"], artefak


if __name__ == "__main__":
    ekstrak_dan_simpan()