"""
================================================================================
MODUL LAPISAN 5 ANFIS: AGREGASI OUTPUT AKHIR & PERHITUNGAN ERROR (RMSE)
Penanggung Jawab : Anggota 5
================================================================================

1. KONSEP & RUMUS MATEMATIS:
   - Lapisan 5 hanya terdiri dari 1 simpul tunggal (simpul penjumlahan) yang
     mengagregasi seluruh sinyal keluaran dari 32 aturan pada Lapisan 4 menjadi
     satu nilai prediksi skalar (masih dalam skala normalisasi [0, 1]):

     y_pred = sum_{r=1}^{32} O_{4,r} = sum_{r=1}^{32} (w_bar_r * f_r)

   - Fungsi evaluasi galat pelatihan menggunakan Root Mean Squared Error (RMSE):
     RMSE = sqrt( (1 / N) * sum_{n=1}^{N} (y_pred_n - y_true_n)^2 )

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - forward_lapis5(o4)
     * Input  : o4 shape (N, 32) dari Lapisan 4
     * Output : y_pred shape (N,) -> vektor 1D berisi hasil prediksi per sampel

   - compute_rmse(y_pred, y_true)
     * Input  : y_pred shape (N,), y_true shape (N,)
     * Output : nilai RMSE (float)

3. INSTRUKSI PENGERJAAN (TODO):
   [x] TODO 1: Pada `forward_lapis5`, jumlahkan seluruh kolom aturan pada `o4`
               menggunakan `np.sum(o4, axis=1)` sehingga menghasilkan array 1D
               berukuran `(N,)`.
   [x] TODO 2: Pada `compute_rmse`, hitung selisih `error = y_pred - y_true`,
               kuadratkan, ambil rata-ratanya (`np.mean`), lalu akarkan (`np.sqrt`),
               dan kembalikan dalam tipe `float`.
================================================================================
"""

import numpy as np


def forward_lapis5(o4):
  # Jumlahkan seluruh keluaran Lapisan 4 menjadi output prediksi tunggal
  y_pred = np.sum(o4, axis=1)
  return y_pred


def compute_rmse(y_pred, y_true):
  # Hitung Root Mean Squared Error (RMSE) antara y_pred dan y_true
  error = y_pred - y_true
  rmse = np.sqrt(np.mean(error ** 2))
  return float(rmse)
