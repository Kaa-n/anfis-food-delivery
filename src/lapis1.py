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


def gaussian_mf(x, c, sigma):
  # TODO: Tulis kode rumus Gaussian di sini
  pass


def forward_lapis1(X, c, sigma):
  # TODO: Tulis kode fuzzifikasi untuk seluruh sampel dan input di sini
  pass


def backward_lapis1(X, mu, w, f, y_pred, y_true, rules, c, sigma, lr):
  # TODO: Tulis kode Gradient Descent untuk memperbarui c dan sigma di sini
  pass