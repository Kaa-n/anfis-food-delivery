"""
================================================================================
MODUL PREPROCESSING DATA: CLEANING, ORDINAL ENCODING & MIN-MAX SCALING
Penanggung Jawab : Anggota 1 (Pendamping Lapisan 1)
================================================================================

1. KONSEP & ALUR PEMROSESAN DATA:
   - Dataset mentah (Food_Delivery_Times.csv) memiliki kombinasi fitur numerik dan
     kategorikal serta sejumlah nilai kosong (NaN).
   - Kolom kategorikal diubah menjadi skala numerik berurutan (Ordinal Encoding):
     * Traffic_Level : Low = 1.0, Medium = 2.0, High = 3.0
     * Weather       : Clear = 1.0, Windy = 2.0, Foggy = 3.0, Rainy = 4.0, Snowy = 5.0
   - Seluruh variabel (5 input & 1 target) dinormalisasi ke rentang [0, 1] dengan
     rumus Min-Max Scaling agar pelatihan ANFIS stabil:

     x_norm = (x - x_min) / (x_max - x_min + 1e-8)

   - Hasil prediksi ANFIS (skala [0, 1]) dikembalikan ke satuan menit dengan rumus
     denormalisasi:

     y_menit = y_norm * (y_max - y_min) + y_min

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - generate_fallback_dataset(n_samples=600, seed=42)
     * Output : pd.DataFrame dengan 6 kolom wajib (FEATURE_COLS + [TARGET_COL])

   - preprocess_data(df_raw, test_size=0.2, random_state=42)
     * Input  : df_raw (pd.DataFrame mentah dari CSV)
     * Output : tuple (df_clean, X_train, X_test, y_train, y_test, scaler_info)
                di mana scaler_info adalah dict:
                {"x_min": array(5,), "x_max": array(5,), "y_min": float, "y_max": float}

   - normalize_single_input(raw_values, scaler_info)
     * Input  : list/array 5 angka mentah, dict scaler_info
     * Output : np.ndarray 1D shape (5,) dalam rentang [0, 1]

   - denormalize_output(y_norm, scaler_info)
     * Input  : float atau np.ndarray dalam rentang [0, 1], dict scaler_info
     * Output : float atau np.ndarray dalam satuan asli (menit)

3. INSTRUKSI PENGERJAAN (TODO):
   [ ] TODO 1: Pada `generate_fallback_dataset`, bangkitkan data simulasi realistis
               menggunakan `np.random.default_rng(seed)` dengan kolom sesuai
               FEATURE_COLS dan TARGET_COL sebagai cadangan jika CSV belum diunggah.
   [ ] TODO 2: Pada `preprocess_data`, validasi keberadaan 6 kolom wajib. Jika ada
               kolom yang hilang, lempar `ValueError`.
   [ ] TODO 3: Lakukan mapping pada kolom `Traffic_Level` dan `Weather` memakai
               TRAFFIC_MAP dan WEATHER_MAP (jika bertipe object/string), konversi
               seluruh kolom ke numerik (`pd.to_numeric`), lalu hapus baris NaN
               dengan `.dropna().reset_index(drop=True)`.
   [ ] TODO 4: Hitung dan simpan `x_min`, `x_max`, `y_min`, `y_max` ke dalam dict
               `scaler_info`, normalisasi X dan y ke [0, 1], lalu bagi menjadi
               80% data latih dan 20% data uji memakai `train_test_split`.
   [ ] TODO 5: Implementasikan rumus normalisasi input tunggal pada
               `normalize_single_input` dan denormalisasi pada `denormalize_output`.
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Traffic_Level",
    "Weather",
]
TARGET_COL = "Delivery_Time_min"

TRAFFIC_MAP = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
WEATHER_MAP = {
    "Clear": 1.0,
    "Windy": 2.0,
    "Foggy": 3.0,
    "Rainy": 4.0,
    "Snowy": 5.0,
}


def generate_fallback_dataset(n_samples=600, seed=42):
  # TODO: Bangkitkan DataFrame simulasi 5 fitur + 1 target Delivery_Time_min
  pass


def preprocess_data(df_raw, test_size=0.2, random_state=42):
  # TODO: Lakukan cleaning, encoding, Min-Max scaling, dan train_test_split 80:20
  pass


def normalize_single_input(raw_values, scaler_info):
  # TODO: Normalisasi 1 baris input baru [dist, prep, exp, traffic, weather] ke [0, 1]
  pass


def denormalize_output(y_norm, scaler_info):
  # TODO: Kembalikan nilai prediksi normalisasi [0, 1] ke satuan menit
  pass