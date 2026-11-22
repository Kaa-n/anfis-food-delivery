"""
Lapisan 4 ANFIS — Sugeno orde-1 dan Least Squares Estimator (Anggota 4).

Kontrak: X (N, 5), w_bar (N, 32), consequents (32, 6).
Kolom konsekuen: p1, p2, p3, p4, p5, q (bias terakhir).
LSE meminimalkan ||A theta - y||² + reg ||theta||², termasuk bias.
"""
import numpy as np


def _validate_inputs(X, w_bar):
    X = np.asarray(X, dtype=float)
    w_bar = np.asarray(w_bar, dtype=float)
    if X.ndim != 2 or X.shape[1] != 5 or X.shape[0] == 0:
        raise ValueError("X harus memiliki shape (N, 5), dengan N > 0.")
    if w_bar.shape != (X.shape[0], 32):
        raise ValueError("w_bar harus memiliki shape (N, 32).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(w_bar)):
        raise ValueError("X dan w_bar tidak boleh mengandung NaN atau infinity.")
    if np.any(w_bar < 0) or np.any(w_bar > 1):
        raise ValueError("w_bar harus berada pada rentang [0, 1].")
    # Tidak menormalisasi ulang: Lapisan 3 menggunakan epsilon dan dapat
    # menghasilkan baris nol ketika seluruh firing strength mengalami underflow.
    return X, w_bar


def forward_lapis4(X, w_bar, consequents):
    """Kembalikan f_r = sum(p_ri*x_i) + q_r dan o4_r = w_bar_r*f_r."""
    X, w_bar = _validate_inputs(X, w_bar)
    consequents = np.asarray(consequents, dtype=float)
    if consequents.shape != (32, 6):
        raise ValueError("consequents harus memiliki shape (32, 6).")
    if not np.all(np.isfinite(consequents)):
        raise ValueError("consequents tidak boleh mengandung NaN atau infinity.")
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    f = X_bias @ consequents.T
    o4 = w_bar * f
    return f, o4


def update_consequents_lse(X, y, w_bar, reg=1e-3):
    """Optimasi 192 parameter; tidak mengubah array masukan.

    Reg > 0: selesaikan persamaan normal ridge tanpa invers eksplisit.
    Reg = 0: least squares SVD agar desain rank-deficient tetap dapat dilatih.
    """
    X, w_bar = _validate_inputs(X, w_bar)
    y = np.asarray(y, dtype=float)
    if y.shape != (X.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError("y harus finite dan memiliki shape (N,).")
    if np.ndim(reg) != 0 or not np.isfinite(reg) or reg < 0:
        raise ValueError("reg harus berupa skalar finite >= 0.")
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    A = (w_bar[:, :, None] * X_bias[:, None, :]).reshape(X.shape[0], 192)
    if reg == 0:
        theta = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        ATA = A.T @ A + reg * np.eye(A.shape[1])
        ATy = A.T @ y
        try:
            theta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            # Sistem augmented mempertahankan objektif ridge jika solve gagal.
            augmented_A = np.vstack([A, np.sqrt(reg) * np.eye(192)])
            augmented_y = np.concatenate([y, np.zeros(192)])
            theta = np.linalg.lstsq(augmented_A, augmented_y, rcond=None)[0]
    if not np.all(np.isfinite(theta)):
        raise ValueError("Optimasi LSE menghasilkan parameter non-finite.")
    return theta.reshape(32, 6)
