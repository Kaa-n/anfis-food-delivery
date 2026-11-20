from datetime import datetime
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(
    page_title="ANFIS Food Delivery Time Estimator",
    page_icon=":material/delivery_dining:",
    layout="wide",
)

# ==============================================================================
# SAFE IMPORTS DENGAN FALLBACK 
# ==============================================================================
MISSING_MODULES = []

# 1. Modul Context Generator (Anggota 4)
try:
  from src.context_generator import get_current_operational_context
except (ImportError, AttributeError):
  MISSING_MODULES.append("src/context_generator.py (Anggota 4 - Seed Konteks)")

  def get_current_operational_context(custom_dt=None):
    now = custom_dt if custom_dt is not None else datetime.now()
    seed_val = int(now.strftime("%Y%m%d%H"))
    rng = np.random.default_rng(seed_val)
    return {
        "timestamp_str": now.strftime("%d %B %Y, Pukul %H:00"),
        "seed": seed_val,
        "Weather": str(rng.choice(["Clear", "Windy", "Rainy"])),
        "Traffic_Level": str(rng.choice(["Low", "Medium", "High"])),
        "Preparation_Time_min": float(rng.integers(10, 25)),
        "Courier_Experience_yrs": round(float(rng.uniform(1.0, 6.0)), 1),
    }


# 2. Modul Preprocessing (Anggota 2)
try:
  from src.preprocessing import (
      FEATURE_COLS,
      TARGET_COL,
      TRAFFIC_MAP,
      WEATHER_MAP,
      denormalize_output,
      generate_fallback_dataset,
      normalize_single_input,
      preprocess_data,
  )
except (ImportError, AttributeError):
  MISSING_MODULES.append(
      "src/preprocessing.py (Anggota 2 - Preprocessing Data)"
  )
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

  def generate_fallback_dataset(n_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(1.0, 15.0, n_samples)
    prep = rng.uniform(5.0, 25.0, n_samples)
    exp = rng.uniform(0.5, 8.0, n_samples)
    traf = rng.choice(["Low", "Medium", "High"], n_samples)
    weat = rng.choice(["Clear", "Windy", "Foggy", "Rainy", "Snowy"], n_samples)
    y = (
        12.0
        + 2.2 * dist
        + 0.8 * prep
        - 0.5 * exp
        + rng.normal(0, 2.0, n_samples)
    )
    return pd.DataFrame({
        "Distance_km": np.round(dist, 2),
        "Preparation_Time_min": np.round(prep, 0),
        "Courier_Experience_yrs": np.round(exp, 1),
        "Traffic_Level": traf,
        "Weather": weat,
        "Delivery_Time_min": np.round(np.clip(y, 10, 100), 1),
    })

  def preprocess_data(df_raw, test_size=0.2, random_state=42):
    df = df_raw[FEATURE_COLS + [TARGET_COL]].copy()
    if df["Traffic_Level"].dtype == object:
      df["Traffic_Level"] = df["Traffic_Level"].map(TRAFFIC_MAP)
    if df["Weather"].dtype == object:
      df["Weather"] = df["Weather"].map(WEATHER_MAP)
    for col in FEATURE_COLS + [TARGET_COL]:
      df[col] = pd.to_numeric(df[col], errors="coerce")
    df_clean = df.dropna().reset_index(drop=True)
    X = df_clean[FEATURE_COLS].values.astype(float)
    y = df_clean[TARGET_COL].values.astype(float)
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = float(y.min()), float(y.max())
    X_norm = (X - x_min) / (x_max - x_min + 1e-8)
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    split_idx = int(len(df_clean) * (1 - test_size))
    scaler_info = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    return (
        df_clean,
        X_norm[:split_idx],
        X_norm[split_idx:],
        y_norm[:split_idx],
        y_norm[split_idx:],
        scaler_info,
    )

  def normalize_single_input(raw_values, scaler_info):
    arr = np.array(raw_values, dtype=float)
    return (arr - scaler_info["x_min"]) / (
        scaler_info["x_max"] - scaler_info["x_min"] + 1e-8
    )

  def denormalize_output(y_norm, scaler_info):
    return (
        y_norm * (scaler_info["y_max"] - scaler_info["y_min"])
        + scaler_info["y_min"]
    )


# 3. Modul Inti ANFIS (Anggota 3)
try:
  from src.anfis import ANFIS
except (ImportError, AttributeError):
  MISSING_MODULES.append("src/anfis.py (Anggota 3 - Algoritma Inti ANFIS)")

  class ANFIS:
    """Stub ANFIS sementara agar UI Streamlit tetap dapat dijalankan saat src/anfis.py belum diisi."""

    def __init__(self, n_inputs=5, lr=0.05, reg=1e-3):
      self.n_inputs = n_inputs
      self.lr = lr
      self.reg = reg
      self.rules = list(itertools.product([0, 1], repeat=n_inputs))
      self.n_rules = len(self.rules)
      self.c = np.tile(np.array([[0.25, 0.75]]), (n_inputs, 1))
      self.sigma = np.full((n_inputs, 2), 0.25)
      self.init_c = self.c.copy()
      self.init_sigma = self.sigma.copy()
      self.consequents = np.full((self.n_rules, n_inputs + 1), 0.1)
      self.loss_history = []

    @staticmethod
    def gaussian_mf(x, c, sigma):
      return np.exp(-0.5 * ((x - c) / (sigma + 1e-8)) ** 2)

    def fit(self, X, y, epochs=50):
      self.loss_history = [
          float(0.25 * np.exp(-0.04 * e) + 0.05) for e in range(epochs)
      ]
      return self.loss_history

    def forward_layers(self, X):
      n = X.shape[0]
      mu = np.zeros((n, self.n_inputs, 2))
      for i in range(self.n_inputs):
        for j in range(2):
          mu[:, i, j] = self.gaussian_mf(
              X[:, i], self.c[i, j], self.sigma[i, j]
          )
      w = np.ones((n, self.n_rules))
      for r, rule in enumerate(self.rules):
        for i, mf_idx in enumerate(rule):
          w[:, r] *= mu[:, i, mf_idx]
      w_bar = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
      f = np.tile(np.mean(X, axis=1, keepdims=True), (1, self.n_rules))
      o4 = w_bar * f
      y_pred = np.sum(o4, axis=1)
      return mu, w, w_bar, f, o4, y_pred


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
    st.session_state.data_source = (
        "Dataset Simulasi Bawaan (Unggah CSV Kaggle di Tab 2)"
    )

if "data_pack" not in st.session_state:
  try:
    st.session_state.data_pack = preprocess_data(st.session_state.df_raw)
  except Exception:
    st.session_state.data_pack = None

if "model" not in st.session_state:
  st.session_state.model = None


def render_layer_inspection(model, raw_vec, x_norm_single, scaler_info):
  """Menampilkan rincian perhitungan Lapisan 1 sampai Lapisan 5 ANFIS."""
  mu, w, w_bar, f, o4, y_pred_norm = model.forward_layers(x_norm_single)
  pred_minutes = float(denormalize_output(y_pred_norm[0], scaler_info))

  st.markdown(r"**Lapisan 1 — Derajat Keanggotaan Gaussian ($\mu$):**")
  df_l1 = pd.DataFrame({
      "Variabel Input": FEATURE_COLS,
      "Nilai Asli": raw_vec,
      "Nilai Normalisasi [0,1]": np.round(x_norm_single[0], 4),
      "μ Rendah (MF 0)": np.round(mu[0, :, 0], 4),
      "μ Tinggi (MF 1)": np.round(mu[0, :, 1], 4),
  })
  st.dataframe(df_l1, use_container_width=True)

  st.markdown(
      r"**Lapisan 2 s.d. Lapisan 4 — Evaluasi 32 Aturan Sugeno (Diurutkan dari"
      r" Bobot Terbesar):**"
  )
  rule_labels = [
      " & ".join(["Rendah" if b == 0 else "Tinggi" for b in r])
      for r in model.rules
  ]
  df_rules_eval = pd.DataFrame({
      "Aturan": [f"R{idx + 1}" for idx in range(model.n_rules)],
      "Kombinasi Premis (x1..x5)": rule_labels,
      "L2: Firing Strength (w)": np.round(w[0], 6),
      "L3: Normalisasi (w_bar)": np.round(w_bar[0], 6),
      "L4: Konsekuen Linear (f)": np.round(f[0], 4),
      "L4: Kontribusi (w_bar * f)": np.round(o4[0], 6),
  }).sort_values(by="L3: Normalisasi (w_bar)", ascending=False)
  st.dataframe(df_rules_eval, use_container_width=True)

  st.markdown(
      r"**Lapisan 5 — Agregasi Output Akhir:** "
      rf"$\hat{{y}}_{{\text{{norm}}}} = \sum_{{r=1}}^{{32}} \bar{{w}}_r f_r = {y_pred_norm[0]:.4f}$ "
      rf"$\xrightarrow{{\text{{denormalisasi}}}}$ **{pred_minutes:.2f} menit**"
  )


# ==============================================================================
# HEADER UTAMA APLIKASI
# ==============================================================================
st.title(
    ":material/delivery_dining: Estimasi Waktu Tiba Pengantaran Makanan"
    " Berbasis ANFIS"
)
st.caption(
    "Adaptive Neuro-Fuzzy Inference System (5 Input, 2 MF Gaussian, 32 Aturan"
    " Sugeno Orde-1)"
)

if MISSING_MODULES:
  st.info(
      "**Mode Simulasi Antarmuka (Stub Mode) Aktif:** Modul berikut di folder"
      " `src/` masih kosong sehingga aplikasi menjalankan fungsi pengganti"
      " sementara:\n"
      + "\n".join([f"- `{m}`" for m in MISSING_MODULES]),
      icon=":material/construction:",
  )

tabs = st.tabs([
    ":material/home: 1. Beranda & Prediksi Utama",
    ":material/database: 2. Dataset & Preprocessing",
    ":material/model_training: 3. Pelatihan Model (Training)",
    ":material/tune: 4. Simulasi & Inspeksi 5 Lapisan",
    ":material/analytics: 5. Evaluasi Model",
    ":material/menu_book: 6. Informasi Sistem & 32 Aturan",
])

# ==============================================================================
# TAB 1: BERANDA, KONDISI LIVE SEED & PREDIKSI UTAMA
# ==============================================================================
with tabs[0]:
  st.subheader("Deskripsi & Tujuan Sistem")
  st.write(
      "Aplikasi navigasi konvensional umumnya hanya menghitung jarak tempuh di"
      " jalan raya, tetapi belum memperhitungkan interaksi non-linear antara"
      " antrean dapur restoran, pengalaman kurir, kepadatan lalu lintas, dan"
      " cuaca. Sistem ini menerapkan **Adaptive Neuro-Fuzzy Inference System"
      " (ANFIS)** untuk mengestimasi total waktu pengantaran makanan (menit)"
      " sekaligus menjelaskan bobot aturan fuzzy yang memengaruhinya."
  )

  st.divider()

  st.subheader(
      ":material/sensors: Kondisi Operasional Saat Ini (Seed Tanggal & Jam)"
  )
  use_custom_time = st.checkbox(
      "Simulasikan tanggal/jam tertentu (untuk menguji perubahan seed saat demo)"
  )
  if use_custom_time:
    sc1, sc2 = st.columns(2)
    sim_date = sc1.date_input("Pilih Tanggal Simulasi", value=datetime.now())
    sim_hour = sc2.slider(
        "Pilih Jam Simulasi (00–23)",
        min_value=0,
        max_value=23,
        value=datetime.now().hour,
    )
    custom_dt = datetime(sim_date.year, sim_date.month, sim_date.day, sim_hour)
    ctx = get_current_operational_context(custom_dt)
  else:
    ctx = get_current_operational_context()

  st.caption(
      f"Waktu Referensi: **{ctx['timestamp_str']}** | Seed Deterministik:"
      f" `{ctx['seed']}`"
  )

  k1, k2, k3, k4 = st.columns(4)
  k1.metric(":material/partly_cloudy_day: Cuaca Saat Ini", ctx["Weather"])
  k2.metric(":material/traffic: Tingkat Kemacetan", ctx["Traffic_Level"])
  k3.metric(
      ":material/skillet: Estimasi Antrean Dapur",
      f"{ctx['Preparation_Time_min']:.0f} menit",
  )
  k4.metric(
      ":material/badge: Pengalaman Kurir",
      f"{ctx['Courier_Experience_yrs']} tahun",
  )

  st.divider()

  st.subheader(":material/bolt: Prediksi Waktu Pengantaran")
  mode_input = st.radio(
      "Pilih Sumber Parameter Input:",
      [
          "Gunakan Kondisi Saat Ini (Otomatis dari Seed)",
          "Input Manual Penuh (Simulasi Bebas)",
      ],
      horizontal=True,
  )

  if mode_input == "Gunakan Kondisi Saat Ini (Otomatis dari Seed)":
    in_dist_home = st.number_input(
        "Masukkan Jarak Restoran ke Pelanggan (km)",
        min_value=0.1,
        max_value=50.0,
        value=5.5,
        step=0.5,
        key="home_dist_auto",
    )
    in_prep_home = float(ctx["Preparation_Time_min"])
    in_exp_home = float(ctx["Courier_Experience_yrs"])
    in_traffic_home = ctx["Traffic_Level"]
    in_weather_home = ctx["Weather"]
  else:
    hm1, hm2, hm3 = st.columns(3)
    in_dist_home = hm1.number_input(
        "Jarak Pengantaran (km)",
        min_value=0.1,
        max_value=50.0,
        value=5.5,
        step=0.5,
        key="home_dist_man",
    )
    in_prep_home = hm2.number_input(
        "Waktu Persiapan Dapur (menit)",
        min_value=1.0,
        max_value=90.0,
        value=15.0,
        step=1.0,
        key="home_prep_man",
    )
    in_exp_home = hm3.number_input(
        "Pengalaman Kurir (tahun)",
        min_value=0.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
        key="home_exp_man",
    )
    hm4, hm5 = st.columns(2)
    in_traffic_home = hm4.selectbox(
        "Tingkat Kemacetan",
        list(TRAFFIC_MAP.keys()),
        index=1,
        key="home_traf_man",
    )
    in_weather_home = hm5.selectbox(
        "Kondisi Cuaca", list(WEATHER_MAP.keys()), index=0, key="home_weat_man"
    )

  if st.session_state.model is None:
    st.warning(
        "Model ANFIS belum dilatih. Anda dapat melatihnya secara lengkap di"
        " **Tab 3**, atau tekan tombol cepat di bawah ini.",
        icon=":material/warning:",
    )
    if st.button(
        "Latih Model Default Sekarang (50 Epoch)",
        icon=":material/rocket_launch:",
    ):
      if st.session_state.data_pack is not None:
        _, X_tr, _, y_tr, _, _ = st.session_state.data_pack
        quick_model = ANFIS(n_inputs=5, lr=0.05, reg=1e-3)
        quick_model.fit(X_tr, y_tr, epochs=50)
        st.session_state.model = quick_model
        st.rerun()
      else:
        st.error(
            "Dataset belum valid. Periksa Tab 2 terlebih dahulu.",
            icon=":material/error:",
        )
  else:
    _, _, _, _, _, scaler_info = st.session_state.data_pack
    raw_home = [
        in_dist_home,
        in_prep_home,
        in_exp_home,
        TRAFFIC_MAP[in_traffic_home],
        WEATHER_MAP[in_weather_home],
    ]
    x_norm_home = normalize_single_input(raw_home, scaler_info).reshape(1, -1)
    _, _, _, _, _, y_pred_home_norm = st.session_state.model.forward_layers(
        x_norm_home
    )
    pred_home_min = float(denormalize_output(y_pred_home_norm[0], scaler_info))

    res1, res2 = st.columns([1, 2])
    with res1:
      st.metric(
          ":material/timer: Estimasi Waktu Tiba", f"{pred_home_min:.1f} menit"
      )
    with res2:
      if pred_home_min <= 30:
        st.success(
            "**Status: Pengantaran Cepat** — Hambatan jalan dan waktu dapur"
            " tergolong rendah.",
            icon=":material/check_circle:",
        )
      elif pred_home_min <= 55:
        st.info(
            "**Status: Pengantaran Normal** — Waktu pengiriman berada pada"
            " rentang standar.",
            icon=":material/info:",
        )
      else:
        st.warning(
            "**Status: Potensi Keterlambatan Tinggi** — Jarak jauh, antrean"
            " dapur, atau cuaca menghambat pengiriman.",
            icon=":material/warning:",
        )

    with st.expander(
        "Lihat Rincian Komputasi 5 Lapisan ANFIS", icon=":material/search:"
    ):
      render_layer_inspection(
          st.session_state.model, raw_home, x_norm_home, scaler_info
      )

  st.divider()
  with st.expander(
      "Ringkasan Arsitektur 5 Lapisan ANFIS", icon=":material/account_tree:"
  ):
    st.markdown(r"""
            * **Lapisan 1 (Fuzzifikasi):** Mengubah 5 input ter-normalisasi menjadi derajat keanggotaan Gaussian ($\mu_{i,0}$ untuk *Rendah*, $\mu_{i,1}$ untuk *Tinggi*).
            * **Lapisan 2 (Firing Strength):** Mengalikan derajat keanggotaan untuk membentuk $2^5 = 32$ aturan ($w_r = \prod_{i=1}^{5} \mu_{i, j_r}$).
            * **Lapisan 3 (Normalisasi):** Menghitung bobot relatif tiap aturan ($\bar{w}_r = w_r / \sum_{k=1}^{32} w_k$).
            * **Lapisan 4 (Defuzzifikasi Sugeno Orde-1):** Menghitung kontribusi fungsi linear tiap aturan ($\bar{w}_r f_r = \bar{w}_r (p_{r,1}x_1 + \dots + p_{r,5}x_5 + q_r)$).
            * **Lapisan 5 (Output Akhir):** Menjumlahkan seluruh keluaran Lapisan 4 ($\hat{y} = \sum_{r=1}^{32} \bar{w}_r f_r$) lalu didenormalisasi ke satuan menit.
            """)

# ==============================================================================
# TAB 2: DATASET & PREPROCESSING
# ==============================================================================
with tabs[1]:
  st.subheader("Manajemen Dataset & Pra-pemrosesan")
  st.info(
      f"Sumber dataset aktif: **{st.session_state.data_source}**",
      icon=":material/info:",
  )

  uploaded = st.file_uploader(
      "Unggah Dataset CSV (Food_Delivery_Times.csv)", type=["csv"]
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
          "Dataset berhasil diunggah dan divalidasi! Silakan latih ulang"
          " model pada Tab 3.",
          icon=":material/check_circle:",
      )
    except Exception as e:
      st.error(f"Validasi gagal: {e}", icon=":material/error:")

  if st.session_state.data_pack is not None:
    df_clean, X_train, X_test, y_train, y_test, scaler_info = (
        st.session_state.data_pack
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(":material/table_rows: Total Data Bersih", f"{len(df_clean)} baris")
    c2.metric(":material/school: Data Latih (80%)", f"{len(X_train)} baris")
    c3.metric(":material/fact_check: Data Uji (20%)", f"{len(X_test)} baris")

    st.markdown("**Keterangan 5 Variabel Input & 1 Target Output:**")
    st.markdown("""
        * `Distance_km` ($x_1$): Jarak tempuh pengantaran dalam kilometer.
        * `Preparation_Time_min` ($x_2$): Waktu penyiapan pesanan di restoran dalam menit.
        * `Courier_Experience_yrs` ($x_3$): Lama pengalaman kerja kurir dalam tahun.
        * `Traffic_Level` ($x_4$): Tingkat kepadatan lalu lintas (`Low = 1`, `Medium = 2`, `High = 3`).
        * `Weather` ($x_5$): Kondisi cuaca (`Clear = 1`, `Windy = 2`, `Foggy = 3`, `Rainy = 4`, `Snowy = 5`).
        * `Delivery_Time_min` ($y$): Target waktu pengantaran aktual dalam menit.
        """)

    st.markdown("**Pratinjau 10 Baris Data Setelah Cleaning & Encoding:**")
    st.dataframe(df_clean.head(10), use_container_width=True)

    st.markdown("**Statistik Deskriptif Variabel:**")
    st.dataframe(df_clean.describe().T, use_container_width=True)
  else:
    st.error(
        "Data belum tersedia atau tidak memiliki kolom wajib: "
        f"{FEATURE_COLS + [TARGET_COL]}",
        icon=":material/error:",
    )

# ==============================================================================
# TAB 3: PELATIHAN MODEL (TRAINING)
# ==============================================================================
with tabs[2]:
  st.subheader("Konfigurasi & Pelatihan Hybrid Learning ANFIS")

  if st.session_state.data_pack is None:
    st.warning(
        "Pastikan dataset valid pada Tab 2 terlebih dahulu.",
        icon=":material/warning:",
    )
  else:
    df_clean, X_train, X_test, y_train, y_test, scaler_info = (
        st.session_state.data_pack
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    epochs = col_p1.slider(
        "Jumlah Epoch", min_value=10, max_value=200, value=50, step=10
    )
    lr = col_p2.select_slider(
        "Learning Rate (Gradient Descent)",
        options=[0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.05,
    )
    reg = col_p3.select_slider(
        "Regularisasi LSE (Lambda)", options=[1e-4, 1e-3, 1e-2], value=1e-3
    )

    if st.button(
        "Latih Model ANFIS", type="primary", icon=":material/play_arrow:"
    ):
      with st.spinner("Menjalankan Hybrid Learning (LSE + Gradient Descent)..."):
        anfis_model = ANFIS(n_inputs=5, lr=lr, reg=reg)
        anfis_model.fit(X_train, y_train, epochs=epochs)
        st.session_state.model = anfis_model
      st.success(
          "Pelatihan selesai! RMSE Akhir (Skala Normalisasi):"
          f" {anfis_model.loss_history[-1]:.4f}",
          icon=":material/check_circle:",
      )

    if st.session_state.model is not None:
      model = st.session_state.model

      st.markdown("**1. Grafik Konvergensi Training (RMSE per Epoch)**")
      fig_loss, ax_loss = plt.subplots(figsize=(8, 3.2))
      ax_loss.plot(
          range(1, len(model.loss_history) + 1),
          model.loss_history,
          marker="o",
          markersize=3,
          color="#1f77b4",
      )
      ax_loss.set_xlabel("Epoch")
      ax_loss.set_ylabel("RMSE (Normalized)")
      ax_loss.grid(True, linestyle="--", alpha=0.5)
      st.pyplot(fig_loss)
      plt.close(fig_loss)

      st.markdown(
          "**2. Adaptasi Fungsi Keanggotaan Gaussian (Sebelum vs Sesudah"
          " Training)**"
      )
      x_grid = np.linspace(0, 1, 200)
      fig_mf, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
      for i in range(5):
        ax = axes[i]
        ax.plot(
            x_grid,
            model.gaussian_mf(x_grid, model.init_c[i, 0], model.init_sigma[i, 0]),
            "b--",
            alpha=0.45,
            label="Rendah (Awal)",
        )
        ax.plot(
            x_grid,
            model.gaussian_mf(x_grid, model.init_c[i, 1], model.init_sigma[i, 1]),
            "r--",
            alpha=0.45,
            label="Tinggi (Awal)",
        )
        ax.plot(
            x_grid,
            model.gaussian_mf(x_grid, model.c[i, 0], model.sigma[i, 0]),
            "b-",
            linewidth=2,
            label="Rendah (Akhir)",
        )
        ax.plot(
            x_grid,
            model.gaussian_mf(x_grid, model.c[i, 1], model.sigma[i, 1]),
            "r-",
            linewidth=2,
            label="Tinggi (Akhir)",
        )
        ax.set_title(FEATURE_COLS[i], fontsize=9)
        ax.grid(True, alpha=0.3)
      axes[0].set_ylabel("Derajat Keanggotaan (μ)")
      axes[-1].legend(loc="upper right", fontsize=7)
      st.pyplot(fig_mf)
      plt.close(fig_mf)

# ==============================================================================
# TAB 4: SIMULASI SKENARIO & INSPEKSI 5 LAPISAN
# ==============================================================================
with tabs[3]:
  st.subheader("Simulasi Skenario & Validasi Perhitungan Per Lapisan")

  if st.session_state.model is None:
    st.warning(
        "Silakan latih model ANFIS terlebih dahulu pada Tab 1 atau Tab 3.",
        icon=":material/warning:",
    )
  else:
    model = st.session_state.model
    _, _, _, _, _, scaler_info = st.session_state.data_pack

    c1, c2, c3 = st.columns(3)
    in_dist = c1.number_input(
        "Jarak Restoran - Pelanggan (km)",
        min_value=0.1,
        max_value=50.0,
        value=6.5,
        step=0.5,
        key="sim_dist",
    )
    in_prep = c2.number_input(
        "Waktu Persiapan Makanan (menit)",
        min_value=1.0,
        max_value=90.0,
        value=15.0,
        step=1.0,
        key="sim_prep",
    )
    in_exp = c3.number_input(
        "Pengalaman Kurir (tahun)",
        min_value=0.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
        key="sim_exp",
    )

    c4, c5 = st.columns(2)
    in_traffic_str = c4.selectbox(
        "Tingkat Kemacetan Lalu Lintas",
        list(TRAFFIC_MAP.keys()),
        index=1,
        key="sim_traf",
    )
    in_weather_str = c5.selectbox(
        "Kondisi Cuaca", list(WEATHER_MAP.keys()), index=0, key="sim_weat"
    )

    raw_vec = [
        in_dist,
        in_prep,
        in_exp,
        TRAFFIC_MAP[in_traffic_str],
        WEATHER_MAP[in_weather_str],
    ]
    x_norm_single = normalize_single_input(raw_vec, scaler_info).reshape(1, -1)
    _, _, _, _, _, y_pred_norm = model.forward_layers(x_norm_single)
    pred_minutes = float(denormalize_output(y_pred_norm[0], scaler_info))

    st.metric(
        ":material/timer: Hasil Prediksi Waktu Pengantaran",
        f"{pred_minutes:.2f} menit",
    )
    st.divider()
    render_layer_inspection(model, raw_vec, x_norm_single, scaler_info)

# ==============================================================================
# TAB 5: EVALUASI MODEL
# ==============================================================================
with tabs[4]:
  st.subheader("Evaluasi Performa Model pada Data Uji (20%)")

  if st.session_state.model is None:
    st.warning(
        "Silakan latih model ANFIS terlebih dahulu pada Tab 1 atau Tab 3.",
        icon=":material/warning:",
    )
  else:
    model = st.session_state.model
    _, _, X_test, _, y_test, scaler_info = st.session_state.data_pack

    _, _, _, _, _, y_pred_test_norm = model.forward_layers(X_test)
    y_actual_min = denormalize_output(y_test, scaler_info)
    y_pred_min = denormalize_output(y_pred_test_norm, scaler_info)

    rmse = np.sqrt(mean_squared_error(y_actual_min, y_pred_min))
    mae = mean_absolute_error(y_actual_min, y_pred_min)
    r2 = r2_score(y_actual_min, y_pred_min)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        ":material/square_foot: RMSE (Root Mean Squared Error)",
        f"{rmse:.2f} menit",
    )
    m2.metric(
        ":material/straighten: MAE (Mean Absolute Error)", f"{mae:.2f} menit"
    )
    m3.metric(":material/monitoring: Koefisien Determinasi (R²)", f"{r2:.4f}")

    fig_eval, ax_eval = plt.subplots(figsize=(9, 3.8))
    limit = min(60, len(y_actual_min))
    ax_eval.plot(
        range(limit),
        y_actual_min[:limit],
        label="Waktu Aktual (menit)",
        marker="o",
        markersize=4,
    )
    ax_eval.plot(
        range(limit),
        y_pred_min[:limit],
        label="Prediksi ANFIS (menit)",
        linestyle="--",
        marker="x",
        markersize=4,
    )
    ax_eval.set_title(
        f"Perbandingan Waktu Aktual vs Prediksi ANFIS ({limit} Sampel Uji"
        " Pertama)"
    )
    ax_eval.set_xlabel("Indeks Sampel Uji")
    ax_eval.set_ylabel("Waktu Pengantaran (menit)")
    ax_eval.legend()
    ax_eval.grid(True, alpha=0.4)
    st.pyplot(fig_eval)
    plt.close(fig_eval)

    df_comp = pd.DataFrame({
        "Waktu Aktual (menit)": np.round(y_actual_min, 2),
        "Prediksi ANFIS (menit)": np.round(y_pred_min, 2),
        "Selisih Error Absolut (menit)": np.round(
            np.abs(y_actual_min - y_pred_min), 2
        ),
    })
    st.dataframe(df_comp.head(20), use_container_width=True)

    st.markdown("**Analisis Kualitas & Keterbatasan Sistem:**")
    st.markdown("""
        * **Kualitas Model:** Kombinasi *Least Squares Estimator* dan *Gradient Descent* mampu memetakan hubungan non-linear antar-variabel dengan error rata-rata (MAE) yang rendah dalam satuan menit.
        * **Keterbatasan:** Variabel kategorikal seperti cuaca dan kemacetan masih menggunakan penyederhanaan skala ordinal (`1–3` dan `1–5`), serta belum memperhitungkan faktor mendadak di lapangan seperti kendala teknis kendaraan kurir.
        """)

# ==============================================================================
# TAB 6: INFORMASI SISTEM & 32 ATURAN
# ==============================================================================
with tabs[5]:
  st.subheader("Informasi Sistem, Parameter Akhir & Panduan Penggunaan")
  st.markdown("""
        **Petunjuk Penggunaan Aplikasi:**
        1. **Tab 1 (Beranda):** Pantau kondisi cuaca & kemacetan terkini berbasis *seed* tanggal/jam, lalu lakukan prediksi cepat (otomatis atau manual).
        2. **Tab 2 (Dataset):** Unggah file `Food_Delivery_Times.csv` dari Kaggle untuk memeriksa hasil pembersihan dan statistik data.
        3. **Tab 3 (Training):** Sesuaikan *Epoch* dan *Learning Rate*, latih model, serta amati pergeseran kurva fungsi keanggotaan Gaussian.
        4. **Tab 4 (Inspeksi Lapisan):** Uji skenario bebas dan buka tabel perhitungan tiap lapisan (Lapisan 1–5) sebagai pembanding hitung manual laporan.
        5. **Tab 5 (Evaluasi):** Tinjau nilai RMSE, MAE, dan $R^2$ pada data uji.
        """)

  if st.session_state.model is not None:
    model = st.session_state.model
    st.markdown(
        r"**1. Parameter Premis Gaussian Terlatih ($c$ dan $\sigma$):**"
    )
    df_prem = pd.DataFrame({
        "Variabel Input": FEATURE_COLS,
        "Center Rendah (c0)": np.round(model.c[:, 0], 4),
        "Sigma Rendah (σ0)": np.round(model.sigma[:, 0], 4),
        "Center Tinggi (c1)": np.round(model.c[:, 1], 4),
        "Sigma Tinggi (σ1)": np.round(model.sigma[:, 1], 4),
    })
    st.dataframe(df_prem, use_container_width=True)

    st.markdown(
        r"**2. Matriks Parameter Konsekuen Sugeno Orde-1 ($p_1 \dots p_5, q$)"
        r" untuk 32 Aturan:**"
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
        index=[f"Rule {i + 1}" for i in range(model.n_rules)],
    )
    st.dataframe(df_cons, use_container_width=True)
  else:
    st.info(
        "Latih model terlebih dahulu untuk menampilkan tabel parameter premis"
        " dan 32 aturan konsekuen.",
        icon=":material/info:",
    )