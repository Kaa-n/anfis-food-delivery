"""
================================================================================
MODUL LAPISAN 2 ANFIS: PEMBANGKITAN 32 ATURAN & FIRING STRENGTH (OPERATOR AND)
Penanggung Jawab : Anggota 2
================================================================================

1. KONSEP & RUMUS MATEMATIS:
   - Dengan 5 variabel input dan masing-masing 2 himpunan fuzzy (Rendah=0, Tinggi=1),
     terbentuk 2^5 = 32 kombinasi aturan IF-THEN Sugeno.
   - Setiap simpul pada Lapisan 2 mengalikan derajat keanggotaan dari Lapisan 1
     yang bersesuaian dengan kombinasi aturan ke-r (operator T-norm produk):

     w_r = mu_{1, j1} * mu_{2, j2} * mu_{3, j3} * mu_{4, j4} * mu_{5, j5}

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - generate_rules(n_inputs=5)
     * Input  : n_inputs (int, default=5)
     * Output : list berisi 32 tuple kombinasi indeks MF, contoh:
                [(0,0,0,0,0), (0,0,0,0,1), ..., (1,1,1,1,1)]

   - forward_lapis2(mu, rules)
     * Input  : mu shape (N, 5, 2) dari Lapisan 1, rules (list 32 tuple)
     * Output : w (firing strength) shape (N, 32)

3. INSTRUKSI PENGERJAAN (TODO):
   [ ] TODO 1: Pada `generate_rules`, gunakan `itertools.product([0, 1], repeat=n_inputs)`
               dan konversi hasilnya menjadi `list`.
   [ ] TODO 2: Pada `forward_lapis2`, siapkan matriks `w` bernilai awal 1 (`np.ones`)
               dengan ukuran `(n_samples, len(rules))`.
   [ ] TODO 3: Lakukan perulangan untuk setiap aturan ke-r (`enumerate(rules)`),
               lalu kalikan `w[:, r]` dengan `mu[:, i, mf_idx]` untuk ke-5 input.
   [ ] TODO 4: Kembalikan matriks `w` berukuran `(N, 32)`.
================================================================================
"""

import itertools
import numpy as np


def generate_rules(n_inputs=5):
  # TODO: Bangkitkan 32 kombinasi aturan menggunakan itertools.product
  pass


def forward_lapis2(mu, rules):
  # TODO: Hitung firing strength (perkalian derajat keanggotaan) tiap aturan
  pass