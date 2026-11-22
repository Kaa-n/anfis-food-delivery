# Kontribusi Anggota 4 — Sugeno, LSE, dan Antarmuka

Dokumen ini memuat bagian laporan yang dapat digabungkan dengan laporan kelompok.
Urutan input: jarak, waktu persiapan, pengalaman kurir, lalu lintas, cuaca.
Implementasi berada di src/lapis4.py, tabs/tab3_training.py, dan tabs/tab4_simulasi.py.

## BAB II — Algoritma Hybrid Learning

### Model Sugeno orde-1 dan parameter konsekuen

ANFIS menggunakan lima masukan dan dua fungsi keanggotaan Gaussian per masukan.
Kombinasi tersebut menghasilkan 32 aturan. Setelah Lapisan 3 menghasilkan bobot
ternormalisasi, setiap aturan pada Lapisan 4 menghitung:

$$f_r = p_{r,1}x_1+p_{r,2}x_2+p_{r,3}x_3+p_{r,4}x_4+p_{r,5}x_5+q_r$$

$$O_{4,r}=\bar w_r f_r$$

Semua masukan pada persamaan tersebut telah dinormalisasi. Setiap aturan memiliki
lima koefisien dan satu bias, sehingga terdapat 192 parameter konsekuen.
Lapisan 4 menghasilkan 32 kontribusi per sampel; Lapisan 5 menjumlahkannya.
Nilai linear f dan prediksi tidak secara matematis dibatasi pada interval [0,1],
meskipun data pelatihan dinormalisasi.

### Forward pass: Least Squares Estimator

Pada forward pass, parameter premis c dan sigma dibuat tetap. Untuk sampel n,
dibentuk vektor z_n = [x_n1, x_n2, x_n3, x_n4, x_n5, 1].
Baris matriks desain disusun dengan urutan aturan yang sama seperti Lapisan 2:

$$A_n=[\bar w_{n,1}z_n,\bar w_{n,2}z_n,\ldots,\bar w_{n,32}z_n]$$

Ukuran A adalah N × 192. Parameter disusun sebagai
theta = [p11, ..., p15, q1, p21, ..., p32,5, q32].
Dengan demikian, prediksi seluruh sampel dapat dituliskan sebagai A theta.

Implementasi menggunakan regularisasi ridge untuk mengurangi masalah parameter
yang tidak teridentifikasi ketika aturan sangat berkorelasi atau data terbatas:

$$J(\theta)=\|A\theta-y\|_2^2+\lambda\|\theta\|_2^2$$

$$ (A^T A+\lambda I)\theta=A^Ty $$

Pustaka NumPy menyelesaikan sistem menggunakan numpy.linalg.solve tanpa
menghitung invers eksplisit. Seluruh koefisien, termasuk bias, diregularisasi.
Jika lambda = 0, numpy.linalg.lstsq digunakan agar desain singular tetap
memiliki solusi least squares dengan norma minimum. Jika solve gagal untuk
lambda positif, digunakan sistem augmented [A; sqrt(lambda)I] dengan target
[y; 0], yang mempertahankan objektif ridge yang sama.

### Backward pass: Gradient Descent

Setelah konsekuen diperbarui, parameter konsekuen dibuat tetap dan galat
dipropagasikan untuk memperbarui pusat dan lebar Gaussian:

$$E=\frac{1}{2N}\sum_n(\hat y_n-y_n)^2$$
$$c\leftarrow c-\eta\frac{\partial E}{\partial c},\qquad
\sigma\leftarrow\sigma-\eta\frac{\partial E}{\partial\sigma}$$

Untuk satu sampel, hubungan gradien dengan firing strength adalah:

$$\frac{\partial E_n}{\partial w_r}
=(\hat y-y)\frac{f_r-\hat y}{\sum_k w_k+\epsilon}$$

Implementasi premis berada pada src/lapis1.py milik Anggota 1. Gradien dirata-ratakan,
dibatasi pada [-1,1], kemudian sigma dibatasi pada [0.05,1.0].
LSE dan GD diulang selama jumlah epoch yang dipilih. RMSE dipakai sebagai metrik
monitoring; pembaruan premis menggunakan gradien galat kuadrat, bukan turunan RMSE.
Grafik tidak dijamin menurun pada setiap epoch karena premis berubah melalui GD.

## BAB IV — Implementasi Kode dan Antarmuka

### 4.1 Struktur implementasi

src/anfis.py mengatur alur Lapisan 1 sampai 5 dan siklus hybrid learning.
Lapisan 4 mempertahankan kontrak masukan X (N,5), w_bar (N,32), serta
consequents (32,6). Keluaran forward_lapis4 adalah pasangan f dan o4,
masing-masing (N,32). update_consequents_lse mengembalikan parameter (32,6).
Validasi menolak dimensi salah, data kosong, nilai non-finite, bobot di luar
[0,1], dan regularisasi negatif. Bobot tidak dinormalisasi ulang agar hasil
Lapisan 3 dengan epsilon tetap dipertahankan.

### 4.2 Implementasi konsekuen dan LSE

Kolom satu ditambahkan di sebelah kanan X sebagai bias.
Perkalian X_bias @ consequents.T menghasilkan fungsi linear seluruh aturan.
Perkalian elemen w_bar * f menghasilkan kontribusi Lapisan 4.
Matriks desain menggunakan broadcasting lalu reshape, dengan setiap enam kolom
mewakili satu aturan. Hasil optimasi dikembalikan ke bentuk (32,6).

Pengujian numerik membandingkan ridge dengan solusi sistem augmented independen,
memeriksa contoh manual, desain singular tanpa regularisasi, bobot nol, masukan
tidak valid, dan integrasi dengan pelatihan ANFIS asli. Perintah pengujian:

    python -m unittest discover -s tests -v

### 4.3 Antarmuka konfigurasi pelatihan (Tab 3)

Pengguna memilih jumlah epoch, learning rate GD, serta lambda LSE.
Tombol pelatihan membuat model baru dan menyimpannya ke session state hanya
sesudah pelatihan berhasil serta riwayat galat valid.
Kesalahan validasi atau numerik ditampilkan agar pengguna dapat memperbaiki data.
Grafik menampilkan RMSE ternormalisasi per epoch. Unduhan CSV menyediakan
pasangan epoch dan RMSE untuk dokumentasi.

Dalam orkestrator saat ini, loss_history dicatat sesudah LSE dan sebelum GD.
Karena GD masih dilakukan pada epoch terakhir, RMSE riwayat terakhir dapat
berbeda dari RMSE model tersimpan. Tab 3 menampilkan kedua nilai dengan label
yang berbeda. Model akhir memakai parameter premis setelah pembaruan terakhir.
Grafik Gaussian sebelum dan sesudah pelatihan tetap tersedia.

### 4.4 Simulasi masukan (Tab 4)

Mode Input manual menyediakan lima variabel sesuai urutan model. Mode Sampel
data uji memilih satu baris X_test secara langsung agar normalisasi sampel tidak
diubah. Nilai asli pada mode tersebut direkonstruksi dari rentang scaler.
Prediksi ditampilkan dalam menit melalui fungsi denormalisasi proyek.
Masukan di luar rentang normalisasi mendapat pemberitahuan ekstrapolasi.
Kondisi bobot aturan hampir nol juga ditandai.

### 4.5 Inspeksi lapisan dan validasi manual

Tabel bersama menampilkan keanggotaan, firing strength, normalisasi, nilai
konsekuen, kontribusi aturan, serta agregasi. Bagian khusus Lapisan 4 menyediakan
pilihan R1–R32, lima perkalian p_i*x_i, bias, jumlah f_r, dan perkalian w_bar*f_r.
Unduhan CSV mempertahankan presisi numerik tanpa pembulatan tampilan, termasuk
input asli dan ternormalisasi, MF, kombinasi aturan, parameter, dan output.
Gunakan ID aturan untuk mencocokkan hasil karena tabel bersama diurutkan
berdasarkan bobot, sedangkan CSV disusun berdasarkan R1–R32.

## Hitung manual Lapisan 4 — contoh pedagogis

Angka berikut adalah contoh sintetis untuk memeriksa aritmetika, bukan hasil
pelatihan dataset Food Delivery. Dua aturan aktif dipakai agar mudah dihitung;
30 aturan lain diberi bobot nol.

x = [0.50, 0.25, 0.40, 0.50, 0.00].

| Aturan | p1 | p2 | p3 | p4 | p5 | q | w_bar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 0.20 | 0.30 | -0.10 | 0.15 | 0.05 | 0.10 | 0.60 |
| R2 | 0.10 | 0.40 | -0.05 | 0.20 | 0.10 | 0.05 | 0.40 |

f1 = 0.20(0.50) + 0.30(0.25) - 0.10(0.40) + 0.15(0.50)
+ 0.05(0.00) + 0.10 = 0.310.

O4,1 = 0.60 × 0.310 = 0.186.

f2 = 0.10(0.50) + 0.40(0.25) - 0.05(0.40) + 0.20(0.50)
+ 0.10(0.00) + 0.05 = 0.280.

O4,2 = 0.40 × 0.280 = 0.112.

Kontribusi diteruskan ke Lapisan 5: 0.186 + 0.112 = 0.298.
Bila rentang target ilustratif 10–100 menit, hasil denormalisasi
adalah 10 + 0.298(100−10) = 36.82 menit.

## Prosedur estafet menggunakan sampel proyek

1. Latih model di Tab 3 dan simpan konfigurasi serta sumber dataset.
2. Di Tab 4, pilih Sampel data uji dan nomor sampel yang disepakati.
3. Unduh CSV untuk mengunci input, parameter, dan bobot pada saat tersebut.
4. Terima w_bar dari Anggota 3, lalu hitung lima produk dan bias setiap aturan.
5. Cocokkan f_r dan O4,r dengan CSV, gunakan toleransi 1e-8 sebelum pembulatan.
6. Serahkan 32 nilai O4 kepada Anggota 5 untuk agregasi dan denormalisasi.

Jangan melatih ulang model di tengah estafet karena parameter konsekuen dan
premis akan berubah. Hasil evaluasi dataset akhir dan tangkapan layar untuk
laporan kelompok harus diambil dari sesi yang sama; contoh sintetis di atas
tidak boleh disebut sebagai hasil pengujian dataset.
