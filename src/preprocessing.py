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

TRAFFIC_MAP = {
    "Low": 1.0,
    "Medium": 2.0,
    "High": 3.0,
}

WEATHER_MAP = {
    "Clear": 1.0,
    "Windy": 2.0,
    "Foggy": 3.0,
    "Rainy": 4.0,
    "Snowy": 5.0,
}

EPSILON = 1e-8


# ==============================================================================
# 1. GENERATE FALLBACK DATASET
# ==============================================================================

def generate_fallback_dataset(n_samples=600, seed=42):
    """
    Menghasilkan dataset simulasi food delivery sebagai dataset cadangan.

    Parameters
    ----------
    n_samples : int
        Jumlah data simulasi yang ingin dihasilkan.

    seed : int
        Seed untuk memastikan hasil random dapat direproduksi.

    Returns
    -------
    pd.DataFrame
        Dataset dengan 5 fitur input dan 1 target.
    """

    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool):
        raise ValueError("n_samples harus berupa bilangan bulat.")

    if n_samples < 2:
        raise ValueError("n_samples minimal 2.")

    rng = np.random.default_rng(seed)

    # 1. Membuat fitur numerik
    distance = np.round(
        rng.uniform(0.8, 20.0, n_samples), 2
    )

    preparation_time = np.round(
        rng.uniform(5.0, 30.0, n_samples), 0
    )

    courier_experience = np.round(
        rng.uniform(0.0, 10.0, n_samples), 1
    )

    # 2. Membuat fitur kategorikal Traffic Level
    traffic = rng.choice(
        ["Low", "Medium", "High"],
        size=n_samples,
        p=[0.4, 0.4, 0.2],
    )

    # 3. Membuat fitur kategorikal Weather
    weather = rng.choice(
        ["Clear", "Windy", "Foggy", "Rainy", "Snowy"],
        size=n_samples,
        p=[0.50, 0.15, 0.10, 0.20, 0.05],
    )

    # 4. Mengubah kategori menjadi angka untuk perhitungan target
    traffic_numeric = np.array(
        [TRAFFIC_MAP[value] for value in traffic],
        dtype=float,
    )

    weather_numeric = np.array(
        [WEATHER_MAP[value] for value in weather],
        dtype=float,
    )

    # 5. Menambahkan noise agar data lebih bervariasi
    noise = rng.normal(
        loc=0.0,
        scale=3.0,
        size=n_samples,
    )

    # 6. Membentuk target Delivery Time
    delivery_time = (
        8.0
        + 2.6 * distance
        + 0.95 * preparation_time
        - 0.65 * courier_experience
        + 3.2 * (traffic_numeric - 1)
        + 2.4 * (weather_numeric - 1)
        + noise
    )

    # Menjaga target tetap positif dan masuk akal
    delivery_time = np.clip(
        delivery_time,
        10.0,
        140.0,
    )

    # 7. Membentuk DataFrame
    df = pd.DataFrame(
        {
            "Distance_km": distance,
            "Preparation_Time_min": preparation_time,
            "Courier_Experience_yrs": courier_experience,
            "Traffic_Level": traffic,
            "Weather": weather,
            "Delivery_Time_min": np.round(delivery_time, 0),
        }
    )

    return df


# ==============================================================================
# 2. PREPROCESSING DATA
# ==============================================================================

def preprocess_data(df_raw, test_size=0.2, random_state=42):
    """
    Melakukan cleaning, ordinal encoding, Min-Max scaling,
    dan pembagian data training serta testing.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Dataset mentah.

    test_size : float
        Proporsi data testing. Default 0.2.

    random_state : int
        Seed pembagian data agar hasil konsisten.

    Returns
    -------
    tuple
        (
            df_clean,
            X_train,
            X_test,
            y_train,
            y_test,
            scaler_info
        )
    """

    # --------------------------------------------------------------------------
    # 1. Validasi tipe data
    # --------------------------------------------------------------------------

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw harus berupa pandas DataFrame.")

    # --------------------------------------------------------------------------
    # 2. Validasi kolom wajib
    # --------------------------------------------------------------------------

    required_cols = FEATURE_COLS + [TARGET_COL]

    missing_cols = [
        col for col in required_cols
        if col not in df_raw.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing_cols}"
        )

    # --------------------------------------------------------------------------
    # 3. Mengambil enam kolom yang dibutuhkan
    # --------------------------------------------------------------------------

    df_clean = df_raw[required_cols].copy()

    # --------------------------------------------------------------------------
    # 4. Ordinal Encoding Traffic Level
    # --------------------------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(df_clean["Traffic_Level"])
        or isinstance(df_clean["Traffic_Level"].dtype, pd.CategoricalDtype)
        or isinstance(df_clean["Traffic_Level"].dtype, pd.StringDtype)
    ):
        df_clean["Traffic_Level"] = (
            df_clean["Traffic_Level"]
            .str.strip()
            .map(TRAFFIC_MAP)
        )

    # --------------------------------------------------------------------------
    # 5. Ordinal Encoding Weather
    # --------------------------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(df_clean["Weather"])
        or isinstance(df_clean["Weather"].dtype, pd.CategoricalDtype)
        or isinstance(df_clean["Weather"].dtype, pd.StringDtype)
    ):
        df_clean["Weather"] = (
            df_clean["Weather"]
            .str.strip()
            .map(WEATHER_MAP)
        )

    # --------------------------------------------------------------------------
    # 6. Konversi seluruh kolom menjadi numerik
    # --------------------------------------------------------------------------

    for col in required_cols:
        df_clean[col] = pd.to_numeric(
            df_clean[col],
            errors="coerce",
        )

    # --------------------------------------------------------------------------
    # 7. Menghapus baris yang memiliki NaN
    # --------------------------------------------------------------------------

    df_clean = (
        df_clean
        .dropna()
        .reset_index(drop=True)
    )

    if df_clean.empty:
        raise ValueError(
            "Dataset kosong setelah proses cleaning. "
            "Periksa nilai NaN dan kategori yang digunakan."
        )

    # --------------------------------------------------------------------------
    # 8. Memisahkan fitur X dan target y
    # --------------------------------------------------------------------------

    X = df_clean[FEATURE_COLS].to_numpy(dtype=float)

    y = df_clean[TARGET_COL].to_numpy(dtype=float)

    # --------------------------------------------------------------------------
    # 9. Menghitung parameter Min-Max
    # --------------------------------------------------------------------------

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    y_min = float(y.min())
    y_max = float(y.max())

    # --------------------------------------------------------------------------
    # 10. Menyimpan informasi scaler
    # --------------------------------------------------------------------------

    scaler_info = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    # --------------------------------------------------------------------------
    # 11. Normalisasi fitur X
    # --------------------------------------------------------------------------

    X_norm = (
        (X - x_min)
        / (x_max - x_min + EPSILON)
    )

    # --------------------------------------------------------------------------
    # 12. Normalisasi target y
    # --------------------------------------------------------------------------

    y_norm = (
        (y - y_min)
        / (y_max - y_min + EPSILON)
    )

    # --------------------------------------------------------------------------
    # 13. Pembagian data training dan testing
    # --------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm,
        y_norm,
        test_size=test_size,
        random_state=random_state,
    )

    return (
        df_clean,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_info,
    )


# ==============================================================================
# 3. NORMALISASI INPUT BARU
# ==============================================================================

def normalize_single_input(raw_values, scaler_info):
    """
    Melakukan normalisasi terhadap satu data input baru.

    Urutan input:
    [Distance, Preparation Time, Courier Experience,
     Traffic Level, Weather]

    Parameters
    ----------
    raw_values : list atau np.ndarray
        Lima nilai input mentah.

    scaler_info : dict
        Informasi scaler dari preprocess_data().

    Returns
    -------
    np.ndarray
        Array ternormalisasi dengan shape (5,).
    """

    raw_values = np.asarray(
        raw_values,
        dtype=float,
    )

    if raw_values.shape != (5,):
        raise ValueError(
            "raw_values harus memiliki tepat 5 nilai dengan shape (5,)."
        )

    if not np.all(np.isfinite(raw_values)):
        raise ValueError(
            "raw_values tidak boleh mengandung NaN atau infinity."
        )

    x_min = np.asarray(scaler_info["x_min"], dtype=float)
    x_max = np.asarray(scaler_info["x_max"], dtype=float)

    if x_min.shape != (5,) or x_max.shape != (5,):
        raise ValueError(
            "x_min dan x_max harus memiliki shape (5,)."
        )

    if not np.all(np.isfinite(x_min)) or not np.all(np.isfinite(x_max)):
        raise ValueError("Parameter scaler input harus bernilai finite.")

    if np.any(x_max < x_min):
        raise ValueError("x_max tidak boleh lebih kecil daripada x_min.")

    normalized_input = (
        (raw_values - x_min)
        / (x_max - x_min + EPSILON)
    )

    return normalized_input


# ==============================================================================
# 4. DENORMALISASI OUTPUT
# ==============================================================================

def denormalize_output(y_norm, scaler_info):
    """
    Mengembalikan hasil prediksi dari skala [0,1]
    ke satuan asli berupa menit.

    Parameters
    ----------
    y_norm : float atau np.ndarray
        Nilai prediksi dalam skala normalisasi.

    scaler_info : dict
        Informasi scaler dari preprocess_data().

    Returns
    -------
    float atau np.ndarray
        Hasil prediksi dalam satuan menit.
    """

    y_min = float(scaler_info["y_min"])
    y_max = float(scaler_info["y_max"])

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError("y_min dan y_max harus bernilai finite.")

    if y_max < y_min:
        raise ValueError("y_max tidak boleh lebih kecil daripada y_min.")

    y_norm = np.asarray(y_norm, dtype=float)

    if not np.all(np.isfinite(y_norm)):
        raise ValueError("y_norm tidak boleh mengandung NaN atau infinity.")

    # Rumus denormalisasi
    y_minutes = (
        y_norm * (y_max - y_min)
        + y_min
    )

    # Mempertahankan tipe float untuk input scalar
    if y_minutes.ndim == 0:
        return float(y_minutes)

    return y_minutes
