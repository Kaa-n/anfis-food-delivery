# Hitung Manual Lapisan 4 — Sampel Data Uji Proyek

Sumber: data/Food_Delivery_Times.csv.
SHA-256 dataset: beb5a92973980105af889e1f305346de182ad50f03233cfd38261f6206bdfa91.
Split test_size=0.2, random_state=42; sampel X_test pertama.
Pelatihan: 10 epoch, learning rate 0.05, lambda 0.001.
Data latih: 728; data uji: 183.
Angka ini berasal dari implementasi ANFIS asli, bukan mode fallback.

Input ternormalisasi: [0.3164948451976831, 0.6666666663888888, 0.1111111109876543, 0.4999999975, 0.0].

Aturan yang dijabarkan: R9, kombinasi MF (0, 1, 0, 0, 0).
Koefisien p1–p5 dan q: [0.3706603417763912, 0.286941849700703, -0.17524443715358518, 0.062222010640523816, 0.16896040431605974, -0.058465577966311105].

Perkalian p_i*x_i:
[0.11731208749143925, 0.19129456638742925, -0.019471604106541012, 0.03111100516470688, 0.0].

f_R9 = (0.117312087491) + (0.191294566387) + (-0.019471604107) + (0.031111005165) + (0.000000000000)
+ (-0.058465577966) = 0.261780476971.

O4_R9 = 0.302542207655 × 0.261780476971
= 0.079199643424.

Selisih perhitungan skalar dengan output program: 0.000e+00.

Seluruh 32 aturan tersedia pada [CSV hitung manual](hitung_manual_lapis4.csv).
Jumlah kontribusi untuk diserahkan ke Anggota 5: 0.311446887021.
Rentang target scaler: 8.0–141.0 menit.

Catatan: preprocessing proyek saat ini menghitung rentang Min-Max sebelum split.
Ini mengikuti kode anggota terkait; evaluasi akhir perlu meninjau potensi
kebocoran informasi rentang data uji. Dokumen ini membuktikan aritmetika Lapisan 4,
bukan menyimpulkan kualitas generalisasi model.
