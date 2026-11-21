"""
Komponen bersama untuk antarmuka Streamlit ANFIS:
1. Custom CSS agar tampilan menyerupai wireframe Streamlit wide-layout (max-width 1200px).
2. Adapter & fallback otomatis (stub mode) agar app.py tetap berjalan lancar
   meskipun modul di folder src/ (lapis1.py - lapis5.py, preprocessing.py,
   context_generator.py) masih dalam tahap pengerjaan oleh anggota tim.
3. Komponen visualisasi inspeksi perhitungan Lapisan 1 sampai Lapisan 5.
"""

from datetime import datetime
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

# ==============================================================================
# KONSTANTA & PEMETAAN VARIABEL GLOBAL
# ==============================================================================
DEFAULT_FEATURE_COLS = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Traffic_Level",
    "Weather",
]
DEFAULT_TARGET_COL = "Delivery_Time_min"

DEFAULT_TRAFFIC_MAP = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
DEFAULT_WEATHER_MAP = {
    "Clear": 1.0,
    "Windy": 2.0,
    "Foggy": 3.0,
    "Rainy": 4.0,
    "Snowy": 5.0,
}


# ==============================================================================
# FALLBACK / STUB IMPLEMENTATIONS (DIGUNAKAN JIKA SRC/ BELUM DIISI)
# ==============================================================================
def _fallback_context(custom_dt=None):
    now = custom_dt if custom_dt is not None else datetime.now()
    seed_value = int(now.strftime("%Y%m%d%H"))
    rng = np.random.default_rng(seed_value)

    hour = now.hour
    if (11 <= hour <= 14) or (17 <= hour <= 20):
        traffic_probs = [0.15, 0.45, 0.40]
        prep_base = rng.uniform(15.0, 28.0)
    elif 6 <= hour <= 22:
        traffic_probs = [0.45, 0.40, 0.15]
        prep_base = rng.uniform(8.0, 18.0)
    else:
        traffic_probs = [0.75, 0.20, 0.05]
        prep_base = rng.uniform(5.0, 12.0)

    traffic = str(rng.choice(["Low", "Medium", "High"], p=traffic_probs))
    weather = str(
        rng.choice(
            ["Clear", "Windy", "Foggy", "Rainy", "Snowy"],
            p=[0.50, 0.20, 0.10, 0.18, 0.02],
        )
    )
    prep_time = round(float(prep_base), 0)
    courier_exp = round(float(rng.uniform(1.0, 7.5)), 1)

    bulan_id = [
        "",
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember",
    ]
    ts_str = f"{now.day} {bulan_id[now.month]} {now.year}, Pukul {now.hour:02d}:00"

    return {
        "timestamp_str": ts_str,
        "seed": seed_value,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": courier_exp,
    }


def _fallback_generate_dataset(n_samples=600, seed=42):
    rng = np.random.default_rng(seed)
    dist = np.round(rng.uniform(0.8, 20.0, n_samples), 2)
    prep = np.round(rng.uniform(5.0, 30.0, n_samples), 0)
    exp = np.round(rng.uniform(0.0, 10.0, n_samples), 1)
    traffic = rng.choice(["Low", "Medium", "High"], size=n_samples, p=[0.4, 0.4, 0.2])
    weather = rng.choice(
        ["Clear", "Windy", "Foggy", "Rainy", "Snowy"],
        size=n_samples,
        p=[0.5, 0.15, 0.1, 0.2, 0.05],
    )

    t_num = np.array([DEFAULT_TRAFFIC_MAP[t] for t in traffic])
    w_num = np.array([DEFAULT_WEATHER_MAP[w] for w in weather])
    noise = rng.normal(0, 3.0, n_samples)
    delivery = np.clip(
        8.0 + 2.6 * dist + 0.95 * prep - 0.65 * exp + 3.2 * (t_num - 1) + 2.4 * (w_num - 1) + noise,
        10.0,
        140.0,
    )
    return pd.DataFrame(
        {
            "Distance_km": dist,
            "Preparation_Time_min": prep,
            "Courier_Experience_yrs": exp,
            "Traffic_Level": traffic,
            "Weather": weather,
            "Delivery_Time_min": np.round(delivery, 0),
        }
    )


def _fallback_preprocess_data(df_raw, test_size=0.2, random_state=42):
    required = DEFAULT_FEATURE_COLS + [DEFAULT_TARGET_COL]
    missing = [col for col in required if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df_raw[required].copy()
    if df["Traffic_Level"].dtype == object:
        df["Traffic_Level"] = df["Traffic_Level"].map(DEFAULT_TRAFFIC_MAP)
    if df["Weather"].dtype == object:
        df["Weather"] = df["Weather"].map(DEFAULT_WEATHER_MAP)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    X = df[DEFAULT_FEATURE_COLS].to_numpy(dtype=float)
    y = df[DEFAULT_TARGET_COL].to_numpy(dtype=float)

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    y_min = float(y.min())
    y_max = float(y.max())

    X_norm = (X - x_min) / (x_max - x_min + 1e-8)
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=test_size, random_state=random_state
    )
    scaler_info = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    return df, X_train, X_test, y_train, y_test, scaler_info


def _fallback_normalize_single(raw_values, scaler_info):
    arr = np.asarray(raw_values, dtype=float)
    return (arr - scaler_info["x_min"]) / (scaler_info["x_max"] - scaler_info["x_min"] + 1e-8)


def _fallback_denormalize_output(y_norm, scaler_info):
    return np.asarray(y_norm) * (scaler_info["y_max"] - scaler_info["y_min"]) + scaler_info["y_min"]


class _FallbackANFIS:
    """Implementasi cadangan ANFIS 5 lapisan saat src/lapis1.py - lapis5.py masih kosong."""

    def __init__(self, n_inputs=5, lr=0.05, reg=1e-3):
        self.n_inputs = n_inputs
        self.n_mfs = 2
        self.lr = lr
        self.reg = reg
        self.rules = list(itertools.product([0, 1], repeat=n_inputs))
        self.n_rules = len(self.rules)

        self.c = np.tile(np.array([[0.25, 0.75]]), (n_inputs, 1))
        self.sigma = np.full((n_inputs, 2), 0.25)
        self.init_c = self.c.copy()
        self.init_sigma = self.sigma.copy()

        self.consequents = np.zeros((self.n_rules, n_inputs + 1))
        self.loss_history = []

    @staticmethod
    def gaussian_mf(x, c, sigma):
        return np.exp(-0.5 * ((x - c) / (sigma + 1e-8)) ** 2)

    def forward_layers(self, X):
        n_samples = X.shape[0]
        mu = np.zeros((n_samples, self.n_inputs, 2))
        for i in range(self.n_inputs):
            for j in range(2):
                mu[:, i, j] = self.gaussian_mf(X[:, i], self.c[i, j], self.sigma[i, j])

        w = np.ones((n_samples, self.n_rules))
        for r, rule in enumerate(self.rules):
            for i, mf_idx in enumerate(rule):
                w[:, r] *= mu[:, i, mf_idx]

        w_sum = np.sum(w, axis=1, keepdims=True) + 1e-12
        w_bar = w / w_sum

        X_bias = np.hstack([X, np.ones((n_samples, 1))])
        f = X_bias @ self.consequents.T
        o4 = w_bar * f
        y_pred = np.sum(o4, axis=1)
        return mu, w, w_bar, f, o4, y_pred

    def fit(self, X, y, epochs=50):
        self.loss_history = []
        n_samples = X.shape[0]
        X_bias = np.hstack([X, np.ones((n_samples, 1))])

        for _ in range(epochs):
            _, _, w_bar, _, _, _ = self.forward_layers(X)
            A = (w_bar[:, :, None] * X_bias[:, None, :]).reshape(n_samples, -1)
            ATA = A.T @ A + self.reg * np.eye(A.shape[1])
            ATy = A.T @ y
            theta = np.linalg.solve(ATA, ATy)
            self.consequents = theta.reshape(self.n_rules, self.n_inputs + 1)

            mu, w, w_bar, f, _, y_pred = self.forward_layers(X)
            error = y_pred - y
            rmse = float(np.sqrt(np.mean(error**2)))
            self.loss_history.append(rmse)

            w_sum = np.sum(w, axis=1) + 1e-12
            dE_dw = (error[:, None] * (f - y_pred[:, None])) / w_sum[:, None]

            grad_c = np.zeros_like(self.c)
            grad_sigma = np.zeros_like(self.sigma)

            for i in range(self.n_inputs):
                for j in range(2):
                    matching_rules = [r for r, rule in enumerate(self.rules) if rule[i] == j]
                    dE_dmu = np.zeros(n_samples)
                    for r in matching_rules:
                        other_prod = np.ones(n_samples)
                        for m, mf_idx in enumerate(self.rules[r]):
                            if m != i:
                                other_prod *= mu[:, m, mf_idx]
                        dE_dmu += dE_dw[:, r] * other_prod

                    diff = X[:, i] - self.c[i, j]
                    sig = self.sigma[i, j] + 1e-8
                    dmu_dc = mu[:, i, j] * (diff / (sig**2))
                    dmu_dsigma = mu[:, i, j] * ((diff**2) / (sig**3))

                    grad_c[i, j] = np.mean(dE_dmu * dmu_dc)
                    grad_sigma[i, j] = np.mean(dE_dmu * dmu_dsigma)

            self.c -= self.lr * np.clip(grad_c, -1.0, 1.0)
            self.sigma = np.clip(
                self.sigma - self.lr * np.clip(grad_sigma, -1.0, 1.0), 0.05, 1.0
            )

        return self.loss_history


# ==============================================================================
# RESOLUSI MODUL DINAMIS (OTOMATIS MEMAKAI SRC/ JIKA SUDAH DIIMPLEMENTASIKAN)
# ==============================================================================
STUB_MODULES = []

try:
    from src.context_generator import get_current_operational_context as _src_ctx

    if _src_ctx() is None:
        raise NotImplementedError("context_generator.py masih stub")
    get_current_operational_context = _src_ctx
except Exception:
    get_current_operational_context = _fallback_context
    STUB_MODULES.append("src/context_generator.py")

try:
    from src.preprocessing import (
        FEATURE_COLS as _src_fcols,
        TARGET_COL as _src_tcol,
        TRAFFIC_MAP as _src_tmap,
        WEATHER_MAP as _src_wmap,
        denormalize_output as _src_denorm,
        generate_fallback_dataset as _src_gen_df,
        normalize_single_input as _src_norm_single,
        preprocess_data as _src_prep,
    )

    _test_df = _src_gen_df(n_samples=10)
    if _test_df is None or _src_prep(_test_df) is None:
        raise NotImplementedError("preprocessing.py masih stub")

    FEATURE_COLS = _src_fcols
    TARGET_COL = _src_tcol
    TRAFFIC_MAP = _src_tmap
    WEATHER_MAP = _src_wmap
    generate_fallback_dataset = _src_gen_df
    preprocess_data = _src_prep
    normalize_single_input = _src_norm_single
    denormalize_output = _src_denorm
except Exception:
    FEATURE_COLS = DEFAULT_FEATURE_COLS
    TARGET_COL = DEFAULT_TARGET_COL
    TRAFFIC_MAP = DEFAULT_TRAFFIC_MAP
    WEATHER_MAP = DEFAULT_WEATHER_MAP
    generate_fallback_dataset = _fallback_generate_dataset
    preprocess_data = _fallback_preprocess_data
    normalize_single_input = _fallback_normalize_single
    denormalize_output = _fallback_denormalize_output
    STUB_MODULES.append("src/preprocessing.py")

try:
    from src.anfis import ANFIS as _SrcANFIS

    _probe = _SrcANFIS(n_inputs=5)
    ANFISClass = _SrcANFIS
except Exception:
    ANFISClass = _FallbackANFIS
    STUB_MODULES.append("src/lapis1.py - lapis5.py")


# ==============================================================================
# CUSTOM CSS & KOMPONEN UI BERSAMA
# ==============================================================================
def inject_custom_css():
    """Menyelaraskan lebar kontainer (max-width 1200px) dan kartu metrik Streamlit."""
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            div[data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #E6EAF1;
                border-radius: 8px;
                padding: 14px 16px;
            }
            div[data-testid="stMetricLabel"] {
                color: #555867;
                font-weight: 500;
            }
            div[data-testid="stMetricValue"] {
                color: #31333F;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_layer_inspection(model, raw_vec, x_norm_single, scaler_info):
    """Menampilkan rincian perhitungan Lapisan 1 sampai Lapisan 5 ANFIS."""
    mu, w, w_bar, f, o4, y_pred_norm = model.forward_layers(x_norm_single)
    pred_minutes = float(denormalize_output(y_pred_norm[0], scaler_info))

    st.markdown(r"**Lapisan 1 — Derajat Keanggotaan Gaussian ($\mu$):**")
    df_l1 = pd.DataFrame(
        {
            "Variabel Input": FEATURE_COLS,
            "Nilai Asli": raw_vec,
            "Nilai Normalisasi [0,1]": np.round(x_norm_single[0], 4),
            "μ Rendah (MF 0)": np.round(mu[0, :, 0], 4),
            "μ Tinggi (MF 1)": np.round(mu[0, :, 1], 4),
        }
    )
    st.dataframe(df_l1, use_container_width=True, hide_index=True)

    st.markdown(
        r"**Lapisan 2 s.d. Lapisan 4 — Evaluasi 32 Aturan Sugeno (Diurutkan dari Bobot Terbesar):**"
    )
    rule_labels = [
        ", ".join(["Rendah" if b == 0 else "Tinggi" for b in r]) for r in model.rules
    ]
    df_rules_eval = pd.DataFrame(
        {
            "Rule": [f"R{idx + 1}" for idx in range(model.n_rules)],
            "Kombinasi Premis": rule_labels,
            "L2 Firing Strength (w)": np.round(w[0], 6),
            "L3 Normalisasi (w_bar)": np.round(w_bar[0], 6),
            "L4 Konsekuen Linear (f)": np.round(f[0], 4),
            "Kontribusi (w_bar * f)": np.round(o4[0], 6),
        }
    ).sort_values(by="L3 Normalisasi (w_bar)", ascending=False)
    st.dataframe(df_rules_eval, use_container_width=True, hide_index=True)

    st.info(
        rf"**Lapisan 5 (Agregasi Akhir):** "
        rf"$\hat{{y}}_{{\text{{norm}}}} = \sum_{{r=1}}^{{32}} \bar{{w}}_r f_r = {y_pred_norm[0]:.4f}$ "
        rf"$\xrightarrow{{\text{{denormalisasi}}}}$ **{pred_minutes:.2f} menit**",
        icon=":material/functions:",
    )