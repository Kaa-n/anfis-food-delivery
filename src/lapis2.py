
"""
================================================================================
MODUL LAPISAN 2 ANFIS: PEMBANGKITAN 32 ATURAN & FIRING STRENGTH
Penanggung Jawab : Anggota 2
================================================================================

Konsep:
- 5 variabel input.
- Setiap input memiliki 2 Membership Function (Rendah=0, Tinggi=1).
- Total aturan = 2^5 = 32.
- Operator AND menggunakan perkalian (Product T-Norm).

Fungsi:
1. generate_rules(n_inputs=5)
2. forward_lapis2(mu, rules)

Input forward:
mu : shape (N, 5, 2)

Output:
w : shape (N, 32)
================================================================================
"""

import itertools
import numpy as np

N_INPUTS = 5
N_MFS = 2


def generate_rules(n_inputs=5):
    """
    Menghasilkan seluruh kombinasi aturan fuzzy.

    Parameters
    ----------
    n_inputs : int
        Jumlah variabel input.

    Returns
    -------
    list
        Daftar kombinasi indeks MF.
        Untuk 5 input menghasilkan 32 aturan.
    """

    if not isinstance(n_inputs, (int, np.integer)) or isinstance(n_inputs, bool):
        raise TypeError("n_inputs harus berupa integer.")

    if n_inputs <= 0:
        raise ValueError("n_inputs harus lebih besar dari 0.")

    rules = list(
        itertools.product([0, 1], repeat=n_inputs)
    )

    return rules


def forward_lapis2(mu, rules):
    """
    Menghitung Firing Strength setiap aturan menggunakan
    operator AND Product T-Norm.

    Parameters
    ----------
    mu : ndarray
        Derajat keanggotaan dari Lapisan 1.
        Shape (N, 5, 2).

    rules : list
        Daftar kombinasi aturan fuzzy.
        Shape (32, 5).

    Returns
    -------
    w : ndarray
        Firing Strength setiap aturan.
        Shape (N, 32).
    """

    mu = np.asarray(mu, dtype=float)
    rules = np.asarray(rules)

    # Validasi input mu
    if mu.ndim != 3:
        raise ValueError("mu harus memiliki 3 dimensi (N, 5, 2).")

    n_samples, n_inputs, n_mfs = mu.shape

    if n_inputs != N_INPUTS or n_mfs != N_MFS:
        raise ValueError("mu harus memiliki shape (N, 5, 2).")

    if not np.all(np.isfinite(mu)):
        raise ValueError("mu mengandung NaN atau infinity.")

    if np.any(mu < 0) or np.any(mu > 1):
        raise ValueError("Derajat keanggotaan harus berada pada rentang 0 sampai 1.")

    # Validasi aturan
    if rules.ndim != 2 or rules.shape[1] != n_inputs:
        raise ValueError("rules harus memiliki shape (jumlah_aturan, 5).")

    if len(rules) == 0:
        raise ValueError("Daftar rules tidak boleh kosong.")

    if not np.all(np.isin(rules, [0, 1])):
        raise ValueError("Indeks MF hanya boleh bernilai 0 atau 1.")

    rules = rules.astype(int)

    # Inisialisasi matriks Firing Strength
    n_rules = len(rules)

    w = np.ones((n_samples, n_rules), dtype=float)

    # Menghitung Product T-Norm
    for r, rule in enumerate(rules):

        for i, mf_idx in enumerate(rule):

            w[:, r] *= mu[:, i, mf_idx]

    return w
