"""
================================================================================
MODUL LAPISAN 3 ANFIS: NORMALISASI FIRING STRENGTH
Penanggung Jawab : Anggota 3
================================================================================

1. KONSEP & RUMUS MATEMATIS:
   - Setiap simpul pada Lapisan 3 menghitung rasio kekuatan penyalaan (firing strength)
     aturan ke-r terhadap total kekuatan seluruh 32 aturan:

     w_bar_r = w_r / (sum_{k=1}^{32} w_k + eps)

   - Sifat wajib: Untuk setiap baris data ke-n, jumlah seluruh w_bar dari r=1 s.d. 32
     harus sama dengan 1.0 (sum(w_bar[n, :]) == 1).

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - forward_lapis3(w)
     * Input  : w (firing strength) shape (N, 32) dari Lapisan 2
     * Output : w_bar (normalized firing strength) shape (N, 32)

3. INSTRUKSI PENGERJAAN (TODO):
   [ ] TODO 1: Hitung jumlah total firing strength per baris sampel menggunakan
               `np.sum(w, axis=1, keepdims=True)`.
   [ ] TODO 2: Tambahkan konstanta kecil `1e-12` pada hasil penjumlahan untuk
               mencegah pembagian dengan nol (ZeroDivisionError / NaN).
   [ ] TODO 3: Bagi matriks `w` dengan total per baris tersebut dan kembalikan
               matriks `w_bar` yang memiliki ukuran tetap `(N, 32)`.
================================================================================
"""

import numpy as np


def forward_lapis3(w):
  # TODO: Hitung normalisasi firing strength tiap baris aturan
  pass