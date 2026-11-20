"""
================================================================================
MODUL GENERATOR KONTEKS OPERASIONAL (SEED TANGGAL & JAM)
Penanggung Jawab : Anggota 2 (Pendamping Lapisan 2)
================================================================================

1. KONSEP & LOGIKA DETERMINISTIK:
   - Halaman utama aplikasi menampilkan kondisi operasional lapangan secara otomatis
     (cuaca, kemacetan, estimasi antrean dapur, dan pengalaman rata-rata kurir)
     tanpa bergantung pada API berbayar.
   - Nilai seed dibentuk dari Tahun, Bulan, Tanggal, dan Jam (format YYYYMMDDHH,
     contoh: 2026112112 untuk 21 November 2026 pukul 12.00) sehingga:
     * Selama masih di jam yang sama, hasil tidak berubah saat halaman di-refresh.
     * Saat berganti jam atau tanggal, kondisi otomatis menyesuaikan pola jam sibuk:
       - Jam sibuk makan siang (11.00-14.00) & makan malam (17.00-20.00):
         peluang kemacetan High & antrean dapur lebih tinggi.
       - Jam normal (06.00-22.00 di luar jam sibuk): lalu lintas dominan Low/Medium.
       - Jam dini hari (23.00-05.00): lalu lintas dominan Low & antrean dapur singkat.

2. KONTRAK INPUT & OUTPUT FUNGSI:
   - get_current_operational_context(custom_dt=None)
     * Input  : custom_dt (objek datetime opsional; jika None gunakan datetime.now())
     * Output : dictionary wajib berisi 6 kunci berikut:
                {
                    "timestamp_str"          : str (contoh: "21 November 2026, Pukul 12:00"),
                    "seed"                   : int (contoh: 2026112112),
                    "Weather"                : str ("Clear"|"Windy"|"Foggy"|"Rainy"|"Snowy"),
                    "Traffic_Level"          : str ("Low"|"Medium"|"High"),
                    "Preparation_Time_min"   : float (contoh: 18.0),
                    "Courier_Experience_yrs" : float (contoh: 3.5),
                }

3. INSTRUKSI PENGERJAAN (TODO):
   [ ] TODO 1: Ambil waktu referensi (`now = custom_dt if custom_dt is not None else datetime.now()`).
   [ ] TODO 2: Ubah waktu tersebut menjadi integer seed `int(now.strftime("%Y%m%d%H"))`
               dan inisialisasi generator acak `rng = np.random.default_rng(seed_value)`.
   [ ] TODO 3: Buat percabangan berdasarkan `now.hour` untuk mengatur probabilitas
               kemacetan (`traffic_probs`) dan rentang waktu masak dapur (`prep_base`).
   [ ] TODO 4: Pilih `Traffic_Level` dan `Weather` menggunakan `rng.choice(...)`,
               serta bangkitkan `Courier_Experience_yrs` menggunakan `rng.uniform(...)`.
   [ ] TODO 5: Kembalikan dictionary dengan 6 nama kunci yang persis sama dengan
               spesifikasi di atas agar langsung terbaca oleh Tab 1 pada app.py.
================================================================================
"""

from datetime import datetime
import numpy as np


def get_current_operational_context(custom_dt=None):
  # TODO: Implementasikan pembangkitan kondisi cuaca, kemacetan, dapur & kurir berbasis seed
  pass