"""
================================================================================
MODUL UTAMA ANFIS: ORKESTRATOR 5 LAPISAN & HYBRID LEARNING
================================================================================

CATATAN INTEGRASI:
- File ini menghubungkan fungsi-fungsi dari:
  - src/lapis1.py (Anggota 1)
  - src/lapis2.py (Anggota 2)
  - src/lapis3.py (Anggota 3)
  - src/lapis4.py (Anggota 4)
  - src/lapis5.py (Anggota 5)

- Selama salah satu file lapis1.py s.d. lapis5.py masih berisi catatan/pass,
  kelas ANFIS di bawah ini akan melemparkan NotImplementedError saat diinisialisasi,
  sehingga app.py otomatis berjalan menggunakan mode simulasi bawaan (fallback).
================================================================================
"""

import numpy as np
from src.lapis1 import backward_lapis1, forward_lapis1, gaussian_mf
from src.lapis2 import forward_lapis2, generate_rules
from src.lapis3 import forward_lapis3
from src.lapis4 import forward_lapis4, update_consequents_lse
from src.lapis5 import compute_rmse, forward_lapis5


class ANFIS:

    def __init__(self, n_inputs=5, lr=0.05, reg=1e-3):
        self.n_inputs = n_inputs
        self.n_mfs = 2
        self.lr = lr
        self.reg = reg

        # 1. Bangkitkan 32 aturan (Anggota 2)
        self.rules = generate_rules(n_inputs)
        if self.rules is None:
            raise NotImplementedError(
                "Fungsi generate_rules di lapis2.py belum diimplementasikan."
            )
        self.n_rules = len(self.rules)

        # 2. Inisialisasi parameter premis Lapisan 1 (Anggota 1)
        self.c = np.tile(np.array([[0.25, 0.75]]), (n_inputs, 1))
        self.sigma = np.full((n_inputs, 2), 0.25)
        self.init_c = self.c.copy()
        self.init_sigma = self.sigma.copy()

        # 3. Inisialisasi parameter konsekuen Lapisan 4 (Anggota 4)
        self.consequents = np.zeros((self.n_rules, n_inputs + 1))
        self.loss_history = []

        # 4. Dry-run validasi: pastikan seluruh Lapisan 1-5 sudah mengembalikan nilai valid
        x_dummy = np.full((2, n_inputs), 0.5)
        y_dummy = np.array([0.4, 0.6])
        mu_d, w_d, w_bar_d, f_d, o4_d, y_pred_d = self.forward_layers(x_dummy)
        if any(out is None for out in (mu_d, w_d, w_bar_d, f_d, o4_d, y_pred_d)):
            raise NotImplementedError(
                "Sebagian fungsi forward di lapis1.py - lapis5.py masih kosong."
            )
        if compute_rmse(y_pred_d, y_dummy) is None:
            raise NotImplementedError(
                "Fungsi compute_rmse di lapis5.py belum diimplementasikan."
            )

    @staticmethod
    def gaussian_mf(x, c, sigma):
        return gaussian_mf(x, c, sigma)

    def forward_layers(self, X):
        """Menjalankan alur maju Lapisan 1 sampai Lapisan 5 secara berurutan."""
        mu = forward_lapis1(X, self.c, self.sigma)          # Lapisan 1 (Anggota 1)
        w = forward_lapis2(mu, self.rules)                  # Lapisan 2 (Anggota 2)
        w_bar = forward_lapis3(w)                           # Lapisan 3 (Anggota 3)
        f, o4 = forward_lapis4(X, w_bar, self.consequents)  # Lapisan 4 (Anggota 4)
        y_pred = forward_lapis5(o4)                         # Lapisan 5 (Anggota 5)
        return mu, w, w_bar, f, o4, y_pred

    def fit(self, X, y, epochs=50):
        """Melatih model ANFIS dengan Hybrid Learning (LSE + Gradient Descent)."""
        self.loss_history = []

        for _ in range(epochs):
            # Tahap 1 (Forward Pass): Hitung Lapisan 1-3, lalu update konsekuen via LSE
            mu = forward_lapis1(X, self.c, self.sigma)
            w = forward_lapis2(mu, self.rules)
            w_bar = forward_lapis3(w)
            self.consequents = update_consequents_lse(X, y, w_bar, reg=self.reg)

            # Evaluasi Lapisan 4-5 & RMSE setelah parameter konsekuen diperbarui
            f, o4 = forward_lapis4(X, w_bar, self.consequents)
            y_pred = forward_lapis5(o4)
            rmse = compute_rmse(y_pred, y)
            self.loss_history.append(rmse)

            # Tahap 2 (Backward Pass): Update parameter premis c & sigma via GD
            self.c, self.sigma = backward_lapis1(
                X, mu, w, f, y_pred, y, self.rules, self.c, self.sigma, self.lr
            )

        # Sinkronisasi akhir konsekuen LSE agar selaras dengan c & sigma hasil GD epoch terakhir
        mu = forward_lapis1(X, self.c, self.sigma)
        w = forward_lapis2(mu, self.rules)
        w_bar = forward_lapis3(w)
        self.consequents = update_consequents_lse(X, y, w_bar, reg=self.reg)

        return self.loss_history