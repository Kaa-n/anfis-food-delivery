"""
================================================================================
MODUL LAPISAN 4 ANFIS: DEFUZZIFIKASI SUGENO ORDE-1 & LEAST SQUARES ESTIMATOR
Penanggung Jawab : Anggota 4
================================================================================

1. KONSEP & RUMUS MATEMATIS:
   - Setiap aturan ke-r memiliki fungsi konsekuen linear Sugeno orde-1:
     f_r = p_{r,1}*x_1 + p_{r,2}*x_2 + p_{r,3}*x_3 + p_{r,4}*x_4 + p_{r,5}*x_5 + q_r

   - Keluaran simpul Lapisan 4 adalah perkalian bobot ternormalisasi dengan f_r:
     O_{4,r} = w_bar_r * f_r

   - Optimasi Forward Pass (Least Squares Estimator / Ridge Regression):
     Karena y_hat = sum(w_bar_r * f_r) bersifat linear terhadap parameter konsekuen
     theta = [p_{1,1}..q_1, ..., p_{32,1}..q_32] (total 32 * 6 = 192 parameter),
     theta dicari secara langsung menggunakan persamaan normal ter-regularisasi:

     theta = (A^T * A + reg * I)^(-1) * A^T * y

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - forward_lapis4(X, w_bar, consequents)
     * Input  : X shape (N, 5), w_bar shape (N, 32), consequents shape (32, 6)
     * Output : tuple (f, o4) di mana f shape (N, 32) dan o4 shape (N, 32)

   - update_consequents_lse(X, y, w_bar, reg=1e-3)
     * Input  : X shape (N, 5), y shape (N,), w_bar shape (N, 32), reg (float)
     * Output : consequents_baru shape (32, 6)

3. INSTRUKSI PENGERJAAN (TODO):
   [ ] TODO 1: Pada `forward_lapis4`, tambahkan kolom bias berisi angka 1 di sebelah
               kanan `X` menggunakan `np.hstack` sehingga menjadi `X_bias` shape (N, 6).
   [ ] TODO 2: Hitung matriks nilai fungsi linear `f = X_bias @ consequents.T`
               (shape (N, 32)), lalu hitung `o4 = w_bar * f` (shape (N, 32)).
   [ ] TODO 3: Pada `update_consequents_lse`, bentuk matriks desain `A` berukuran
               `(N, 32 * 6)` dari perkalian elemen `w_bar` dan `X_bias`:
               `A = (w_bar[:, :, None] * X_bias[:, None, :]).reshape(n_samples, -1)`
   [ ] TODO 4: Hitung `ATA = A.T @ A + reg * np.eye(A.shape[1])` dan `ATy = A.T @ y`,
               lalu selesaikan sistem linear menggunakan `np.linalg.solve(ATA, ATy)`
               dan ubah bentuknya kembali ke `(32, 6)`.
================================================================================
"""

import numpy as np


def forward_lapis4(X, w_bar, consequents):
  # TODO: Hitung f (output fungsi linear Sugeno) dan o4 (w_bar * f)
  pass


def update_consequents_lse(X, y, w_bar, reg=1e-3):
  # TODO: Optimasi parameter konsekuen menggunakan Least Squares Estimator
  pass