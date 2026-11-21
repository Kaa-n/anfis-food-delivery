"""
================================================================================
MODUL LAPISAN 3 ANFIS: NORMALISASI FIRING STRENGTH
Penanggung Jawab : Anggota 3
================================================================================

1. KONSEP & RUMUS MATEMATIS:
   - Forward Pass:
     Setiap simpul pada Lapisan 3 menghitung rasio kekuatan penyalaan (firing
     strength) aturan ke-r terhadap total kekuatan seluruh 32 aturan:

     w_bar_r = w_r / (sum_{k=1}^{32} w_k + eps)

     Sifat wajib: Untuk setiap baris data ke-n, jumlah seluruh w_bar dari r=1
     s.d. 32 harus sama dengan 1.0 (sum(w_bar[n, :]) == 1).

   - Backward Pass (Aturan Rantai / Quotient Rule untuk Gradient Descent):
     Jika dE/dw_bar adalah gradien error dari Lapisan 4 terhadap w_bar, maka
     gradien error terhadap firing strength w_j (sebelum normalisasi) adalah:

     dE/dw_j = (dE/dw_bar_j - sum_{r=1}^{32} (dE/dw_bar_r * w_bar_r)) / (sum_{k=1}^{32} w_k + eps)

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - forward_lapis3(w)
     * Input  : w (firing strength) shape (N, 32) dari Lapisan 2
     * Output : w_bar (normalized firing strength) shape (N, 32)

   - backward_lapis3(dE_dw_bar, w)
     * Input  : dE_dw_bar shape (N, 32) dari Lapisan 4, w shape (N, 32) dari Lapisan 2
     * Output : dE_dw shape (N, 32) untuk diteruskan ke Lapisan 2

3. INSTRUKSI PENGERJAAN (SELESAI):
   [x] TODO 1: Hitung jumlah total firing strength per baris sampel menggunakan
               `np.sum(w, axis=1, keepdims=True)`.
   [x] TODO 2: Tambahkan konstanta kecil `1e-12` pada hasil penjumlahan untuk
               mencegah pembagian dengan nol (ZeroDivisionError / NaN).
   [x] TODO 3: Bagi matriks `w` dengan total per baris tersebut dan kembalikan
               matriks `w_bar` yang memiliki ukuran tetap `(N, 32)`.
   [x] TODO 4: Implementasi `backward_lapis3` serta blok pengujian mandiri.
================================================================================
"""

import numpy as np

# Konstanta epsilon untuk mencegah pembagian dengan nol
_EPSILON = 1e-12


def _validasi_firing_strength(w):
    """Memvalidasi dimensi, tipe numerik, dan rentang nilai non-negatif matriks w."""
    w = np.asarray(w, dtype=float)

    if w.ndim != 2:
        raise ValueError(
            f"w harus memiliki 2 dimensi (N, 32), "
            f"tetapi mendapatkan {w.ndim} dimensi."
        )
    if w.shape[1] != 32:
        raise ValueError(
            f"w harus memiliki 32 kolom (aturan), "
            f"tetapi mendapatkan {w.shape[1]} kolom."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("w mengandung NaN atau nilai infinity.")
    if np.any(w < 0):
        raise ValueError(
            "Firing strength tidak boleh bernilai negatif. "
            "Periksa keluaran Lapisan 2."
        )

    return w


def forward_lapis3(w):
    """
    Lapisan 3 ANFIS: Menormalisasi firing strength setiap aturan.

    Setiap simpul pada lapisan ini menghitung rasio kekuatan penyalaan
    aturan ke-r terhadap total kekuatan seluruh 32 aturan (Normalisasi L1):

        w_bar_r = w_r / (sum_{k=1}^{32} w_k + eps)

    Sifat wajib hasil: sum(w_bar[n, :]) == 1.0 untuk setiap baris ke-n.

    Parameters
    ----------
    w : ndarray
        Firing strength dari Lapisan 2.
        Shape (N, 32), di mana N adalah jumlah sampel dan 32 adalah jumlah aturan.
        Setiap elemen harus bernilai non-negatif (hasil perkalian nilai Gaussian).

    Returns
    -------
    w_bar : ndarray
        Normalized firing strength.
        Shape (N, 32), dengan setiap baris berjumlah 1.0.

    Raises
    ------
    ValueError
        Jika w tidak memiliki shape (N, 32) atau mengandung nilai negatif / NaN.
    """
    w = _validasi_firing_strength(w)

    # Jumlahkan firing strength per baris -> shape (N, 1)
    w_sum = np.sum(w, axis=1, keepdims=True)

    # Tambahkan epsilon untuk mencegah pembagian dengan nol
    w_sum_safe = w_sum + _EPSILON

    # Normalisasi — bagi setiap baris dengan total baris tersebut
    w_bar = w / w_sum_safe

    return w_bar


def backward_lapis3(dE_dw_bar, w):
    """
    Backward Pass Lapisan 3 ANFIS: Menghitung gradien error terhadap firing strength w.

    Berdasarkan turunan fungsi normalisasi (quotient rule):

        dE/dw_j = (dE/dw_bar_j - sum_{r=1}^{32} (dE/dw_bar_r * w_bar_r)) / (sum_{k=1}^{32} w_k + eps)

    Parameters
    ----------
    dE_dw_bar : ndarray
        Gradien error dari Lapisan 4 terhadap normalized firing strength w_bar.
        Shape (N, 32). Pada loss kuadratik (y_pred - y), nilainya adalah
        `(error[:, None] * f)`.
    w : ndarray
        Firing strength asli dari Lapisan 2 saat forward pass.
        Shape (N, 32).

    Returns
    -------
    dE_dw : ndarray
        Gradien error terhadap firing strength w untuk diteruskan ke Lapisan 2.
        Shape (N, 32).

    Raises
    ------
    ValueError
        Jika shape dE_dw_bar tidak sama dengan w, atau mengandung NaN / infinity.
    """
    w = _validasi_firing_strength(w)
    dE_dw_bar = np.asarray(dE_dw_bar, dtype=float)

    if dE_dw_bar.shape != w.shape:
        raise ValueError(
            f"Shape dE_dw_bar {dE_dw_bar.shape} harus sama dengan shape w {w.shape}."
        )
    if not np.all(np.isfinite(dE_dw_bar)):
        raise ValueError("dE_dw_bar mengandung NaN atau nilai infinity.")

    w_sum_safe = np.sum(w, axis=1, keepdims=True) + _EPSILON
    w_bar = w / w_sum_safe

    # Jumlah terbobot gradien masuk per baris sampel -> shape (N, 1)
    weighted_grad_sum = np.sum(dE_dw_bar * w_bar, axis=1, keepdims=True)

    # Gradien terhadap w -> shape (N, 32)
    dE_dw = (dE_dw_bar - weighted_grad_sum) / w_sum_safe

    return dE_dw


if __name__ == "__main__":
    print("=== Pengujian Mandiri Modul Lapisan 3 (src/lapis3.py) ===")
    rng = np.random.default_rng(42)

    # 1. Uji Forward Pass pada data acak (N=5, 32 aturan)
    w_uji = rng.uniform(0.01, 1.0, size=(5, 32))
    w_bar_uji = forward_lapis3(w_uji)
    assert w_bar_uji.shape == (5, 32), "Shape output w_bar tidak sesuai!"
    assert np.allclose(w_bar_uji.sum(axis=1), 1.0), "Total w_bar per baris tidak sama dengan 1.0!"
    print("[PASS] Forward pass valid: shape (5, 32) dan jumlah setiap baris == 1.0")

    # 2. Uji stabilitas numerik saat seluruh firing strength bernilai 0
    w_nol = np.zeros((3, 32))
    w_bar_nol = forward_lapis3(w_nol)
    assert np.all(np.isfinite(w_bar_nol)), "Ditemukan NaN/Inf saat input bernilai nol!"
    print("[PASS] Stabilitas numerik aman saat seluruh firing strength == 0.0")

    # 3. Uji Backward Pass (membandingkan gradien analitik vs numerik / finite difference)
    f_konsekuen = rng.normal(size=(5, 32))
    y_target = rng.normal(size=(5,))

    def hitung_loss(w_in):
        wb = forward_lapis3(w_in)
        y_pred = np.sum(wb * f_konsekuen, axis=1)
        return 0.5 * np.sum((y_pred - y_target) ** 2)

    y_pred_awal = np.sum(w_bar_uji * f_konsekuen, axis=1)
    dE_dw_bar_uji = (y_pred_awal - y_target)[:, None] * f_konsekuen
    grad_analitik = backward_lapis3(dE_dw_bar_uji, w_uji)

    grad_numerik = np.zeros_like(w_uji)
    eps_fd = 1e-7
    for i in range(w_uji.shape[0]):
        for j in range(w_uji.shape[1]):
            w_plus = w_uji.copy()
            w_minus = w_uji.copy()
            w_plus[i, j] += eps_fd
            w_minus[i, j] -= eps_fd
            grad_numerik[i, j] = (hitung_loss(w_plus) - hitung_loss(w_minus)) / (2 * eps_fd)

    max_diff = np.max(np.abs(grad_analitik - grad_numerik))
    assert np.allclose(grad_analitik, grad_numerik, atol=1e-6), "Gradien backward_lapis3 tidak akurat!"
    print(f"[PASS] Backward pass terverifikasi (selisih maksimum vs numerik: {max_diff:.2e})")

    # 4. Uji penanganan error pada input tidak valid
    try:
        forward_lapis3(np.ones((5, 16)))
        assert False, "Seharusnya memunculkan ValueError untuk kolom != 32"
    except ValueError:
        print("[PASS] Validasi dimensi (bukan 32 kolom) memunculkan ValueError dengan benar")