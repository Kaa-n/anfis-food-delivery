"""
Paket inti pemrosesan data dan algoritma Adaptive Neuro-Fuzzy Inference System (ANFIS)
untuk estimasi waktu tiba pengantaran makanan.

Pembagian Modul & Penanggung Jawab:
- preprocessing.py     : Pembersihan data, ordinal encoding & Min-Max scaling (Anggota 1)
- lapis1.py            : Lapisan 1 Fuzzifikasi Gaussian & Backward GD (Anggota 1)
- context_generator.py : Generator kondisi operasional berbasis seed waktu (Anggota 2)
- lapis2.py            : Lapisan 2 Pembangkitan 32 aturan & Firing Strength (Anggota 2)
- lapis3.py            : Lapisan 3 Normalisasi Firing Strength (Anggota 3)
- lapis4.py            : Lapisan 4 Defuzzifikasi Sugeno Orde-1 & LSE (Anggota 4)
- lapis5.py            : Lapisan 5 Agregasi Output Akhir & RMSE (Anggota 5)
- anfis.py             : Orkestrator utama yang menggabungkan Lapisan 1-5
"""