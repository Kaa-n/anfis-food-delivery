"""
================================================================================
MODUL LAPISAN 1 ANFIS: FUZZIFIKASI & PEMBARUAN PARAMETER PREMIS (BACKWARD PASS)
Penanggung Jawab : Anggota 1
================================================================================

1. KONSEP & RUMUS MATEMATIS:
   - Setiap variabel input x_i (dari 5 input) dipetakan ke 2 himpunan fuzzy:
     j = 0 (Rendah) dan j = 1 (Tinggi) menggunakan Fungsi Keanggotaan Gaussian:

     mu_{i,j}(x_i) = exp( -0.5 * ((x_i - c_{i,j}) / (sigma_{i,j} + eps))^2 )

   - Parameter premis yang dilatih:
     * c     : titik pusat (center) kurva Gaussian, shape (5, 2)
     * sigma : lebar sebaran (spread) kurva Gaussian, shape (5, 2)

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - gaussian_mf(x, c, sigma)
     * Input  : x (array 1D shape (N,)), c (float), sigma (float)
     * Output : derajat keanggotaan mu (array 1D shape (N,))

   - forward_lapis1(X, c, sigma)
     * Input  : X shape (N, 5), c shape (5, 2), sigma shape (5, 2)
     * Output : mu shape (N, 5, 2)

   - backward_lapis1(X, mu, w, f, y_pred, y_true, rules, c, sigma, lr)
     * Input  : hasil forward pass (X, mu, w, f, y_pred), target y_true (N,),
                daftar 32 aturan (rules), parameter c & sigma, learning rate (lr)
     * Output : c_baru shape (5, 2), sigma_baru shape (5, 2)

3. INSTRUKSI PENGERJAAN (TODO):
   [ ] TODO 1: Implementasikan rumus kurva Gaussian pada fungsi `gaussian_mf`.
               Tambahkan konstanta kecil (1e-8) pada penyebut agar tidak dibagi nol.
   [ ] TODO 2: Pada `forward_lapis1`, lakukan perulangan untuk 5 input (i=0..4)
               dan 2 MF (j=0..1), lalu simpan hasilnya ke matriks `mu` berukuran (N, 5, 2).
   [ ] TODO 3: Pada `backward_lapis1`, hitung turunan error terhadap c dan sigma
               menggunakan aturan rantai (Gradient Descent):
               - dE/dw_r = (y_pred - y_true) * (f_r - y_pred) / sum(w)
               - dE/dmu_{i,j} = jumlah dari (dE/dw_r * perkalian mu input lain selain i)
                 untuk semua aturan r yang menggunakan MF j pada input i.
               - dmu/dc = mu * ((x - c) / sigma^2)
               - dmu/dsigma = mu * ((x - c)^2 / sigma^3)
   [ ] TODO 4: Lakukan gradient clipping (-1.0 s.d. 1.0), perbarui c dan sigma
               (c = c - lr * grad_c), dan batasi sigma dengan np.clip(..., 0.05, 1.0).
================================================================================
"""

import numpy as np
EPSILON = 1e-8
N_INPUTS = 5
N_MFS = 2
N_RULES = 2 ** N_INPUTS

def gaussian_mf(x, c, sigma):
    """
    Menghitung derajat keanggotaan Gaussian.

    x     : array 1D shape (N,)
    c     : titik pusat Gaussian
    sigma : lebar sebaran Gaussian

    Return: array shape (N,)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x harus berupa array 1 dimensi.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x mengandung NaN atau infinity.")
    if not np.isfinite(c):
        raise ValueError("Parameter c harus finite.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma harus lebih besar dari 0.")

    sigma_safe = float(sigma) + EPSILON
    z = (x - float(c)) / sigma_safe

    mu = np.exp(-0.5 * (z ** 2))

    return mu


def forward_lapis1(X, c, sigma):
    """
    Melakukan fuzzifikasi Gaussian untuk seluruh sampel.

    X     : shape (N, 5)
    c     : shape (5, 2)
    sigma : shape (5, 2)

    Return: mu shape (N, 5, 2)
    """
    X = np.asarray(X, dtype=float)
    c = np.asarray(c, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if X.ndim != 2 or X.shape[1] != N_INPUTS:
        raise ValueError(f"X harus memiliki shape (N, {N_INPUTS}).")
    if c.shape != (N_INPUTS, N_MFS):
        raise ValueError("c harus memiliki shape (5, 2).")
    if sigma.shape != (N_INPUTS, N_MFS):
        raise ValueError("sigma harus memiliki shape (5, 2).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X mengandung NaN atau infinity.")
    if not np.all(np.isfinite(c)):
        raise ValueError("c mengandung NaN atau infinity.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Semua sigma harus lebih besar dari 0.")

    n_samples = X.shape[0]
    mu = np.zeros((n_samples, N_INPUTS, N_MFS))

    for i in range(N_INPUTS):
        for j in range(N_MFS):
            mu[:, i, j] = gaussian_mf(
                X[:, i],
                c[i, j],
                sigma[i, j]
            )

    return mu


def backward_lapis1(X, mu, w, f, y_pred, y_true,
                    rules, c, sigma, lr):
    """
    Memperbarui parameter premis c dan sigma
    menggunakan Gradient Descent.

    Return:
    - c_new     : shape (5, 2)
    - sigma_new : shape (5, 2)
    """

    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    f = np.asarray(f, dtype=float)

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)

    rules_raw = np.asarray(rules)
    c = np.asarray(c, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if X.ndim != 2 or X.shape[1] != N_INPUTS:
        raise ValueError("X harus memiliki shape (N, 5).")

    n_samples = X.shape[0]

    if mu.shape != (n_samples, N_INPUTS, N_MFS):
        raise ValueError("Shape mu tidak sesuai.")
    if w.shape != (n_samples, N_RULES):
        raise ValueError("Shape w tidak sesuai.")
    if f.shape != (n_samples, N_RULES):
        raise ValueError("Shape f tidak sesuai.")
    if y_pred.shape != (n_samples,) or y_true.shape != (n_samples,):
        raise ValueError("y_pred dan y_true harus memiliki shape (N,).")
    if rules_raw.shape != (N_RULES, N_INPUTS):
        raise ValueError("rules harus memiliki shape (32, 5).")
    if not np.all(np.isin(rules_raw, [0, 1])):
        raise ValueError("Elemen rules hanya boleh 0 atau 1.")
    if c.shape != (N_INPUTS, N_MFS) or sigma.shape != (N_INPUTS, N_MFS):
        raise ValueError("c dan sigma harus memiliki shape (5, 2).")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate harus lebih besar dari 0.")

    arrays = [X, mu, w, f, y_pred, y_true, c, sigma]
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("Input mengandung NaN atau infinity.")
    if np.any(sigma <= 0):
        raise ValueError("Sigma harus lebih besar dari 0.")
    if np.any(w < 0):
        raise ValueError("Firing strength tidak boleh negatif.")

    rules_array = rules_raw.astype(int)

    # 1. Menghitung turunan error terhadap firing strength.
    error = y_pred - y_true
    sum_w = np.sum(w, axis=1) + EPSILON

    dE_dw = (
        error[:, np.newaxis] *
        (f - y_pred[:, np.newaxis])
    ) / sum_w[:, np.newaxis]

    grad_c = np.zeros((N_INPUTS, N_MFS))
    grad_sigma = np.zeros((N_INPUTS, N_MFS))

    # 2. Menghitung gradien setiap fungsi keanggotaan.
    for i in range(N_INPUTS):
        for j in range(N_MFS):

            dE_dmu = np.zeros(n_samples)

            for r, rule in enumerate(rules_array):

                # Hanya aturan yang menggunakan MF j
                # pada input ke-i yang dihitung.
                if rule[i] != j:
                    continue

                # Perkalian derajat keanggotaan
                # dari empat input lainnya.
                other_mu = np.ones(n_samples)

                for k in range(N_INPUTS):
                    if k != i:
                        other_mu *= mu[:, k, rule[k]]

                dE_dmu += dE_dw[:, r] * other_mu

            sigma_safe = sigma[i, j] + EPSILON
            difference = X[:, i] - c[i, j]

            # Turunan Gaussian terhadap c dan sigma.
            dmu_dc = (
                mu[:, i, j] * difference /
                (sigma_safe ** 2)
            )

            dmu_dsigma = (
                mu[:, i, j] * (difference ** 2) /
                (sigma_safe ** 3)
            )

            # Rata-rata gradien seluruh sampel.
            grad_c[i, j] = np.mean(dE_dmu * dmu_dc)
            grad_sigma[i, j] = np.mean(dE_dmu * dmu_dsigma)

    # 3. Gradient clipping.
    grad_c = np.clip(grad_c, -1.0, 1.0)
    grad_sigma = np.clip(grad_sigma, -1.0, 1.0)

    # 4. Memperbarui parameter premis.
    c_new = c - lr * grad_c
    sigma_new = sigma - lr * grad_sigma

    # Membatasi sigma antara 0.05 dan 1.0.
    sigma_new = np.clip(sigma_new, 0.05, 1.0)

    return c_new, sigma_new