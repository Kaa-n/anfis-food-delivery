# Estimasi Waktu Tiba Pengantaran Makanan Berbasis ANFIS

Aplikasi cerdas berbasis web untuk mengestimasi total waktu pengantaran makanan (*food delivery time*) menggunakan pendekatan **Adaptive Neuro-Fuzzy Inference System (ANFIS)** dengan antarmuka interaktif **Streamlit**.

Proyek ini disusun untuk memenuhi tugas Final Project pada mata kuliah **Teknologi Logika Fuzzy**, Program Studi Informatika, Universitas Udayana.

## Deskripsi & Arsitektur Sistem

Aplikasi navigasi konvensional umumnya hanya menghitung estimasi waktu tempuh berdasarkan jarak jalan raya, namun sering meleset akibat faktor antrean dapur restoran, kondisi cuaca, dan kepadatan lalu lintas yang bersifat tidak pasti (*fuzzy*) serta non-linear.

Sistem ini memodelkan hubungan non-linear tersebut menggunakan arsitektur **ANFIS 5 Lapisan** dengan konfigurasi berikut:

* **Variabel Input ($5$):**
  1. `Distance_km` ($x_1$): Jarak restoran ke lokasi pelanggan ($\text{km}$).
  2. `Preparation_Time_min` ($x_2$): Estimasi durasi penyiapan makanan di dapur ($\text{menit}$).
  3. `Courier_Experience_yrs` ($x_3$): Lama pengalaman kerja kurir ($\text{tahun}$).
  4. `Traffic_Level` ($x_4$): Tingkat kemacetan lalu lintas (`Low = 1`, `Medium = 2`, `High = 3`).
  5. `Weather` ($x_5$): Kondisi cuaca (`Clear = 1`, `Windy = 2`, `Foggy = 3`, `Rainy = 4`, `Snowy = 5`).

* **Variabel Output ($1$):**
  * `Delivery_Time_min` ($\hat{y}$): Estimasi total waktu pengantaran hingga pesanan tiba ($\text{menit}$).

* **Basis Aturan & Pembelajaran:**
  * Menggunakan $2$ fungsi keanggotaan *Gaussian* (*Rendah* dan *Tinggi*) pada setiap input sehingga membentuk $2^5 = 32$ aturan **Sugeno Orde-1**.
  * Dilatih menggunakan **Hybrid Learning Algorithm** yang menggabungkan *Least Squares Estimator* (LSE) pada *forward pass* untuk mengoptimasi parameter konsekuen dan *Gradient Descent* (GD) pada *backward pass* untuk mengoptimasi parameter premis.

* **Fitur Utama Aplikasi:**
  * **Live Operational Context:** Menghasilkan kondisi cuaca, kemacetan, dan antrean dapur saat ini secara deterministik menggunakan *seed* berbasis tanggal dan jam (`YYYYMMDDHH`), disertai opsi simulasi input manual penuh.
  * **Modular 5-Layer Architecture:** Setiap lapisan ANFIS dipisahkan ke dalam modul tersendiri (`lapis1.py` hingga `lapis5.py`) agar mudah dikembangkan dan diuji secara paralel oleh kelima anggota tim.
  * **Graceful Fallback (Stub Mode):** Antarmuka `app.py` tetap dapat dijalankan secara penuh meskipun modul di folder `src/` masih berisi catatan/kerangka awal selama tahap pengembangan.
  * **Visualisasi Adaptasi Kurva:** Menampilkan grafik perbandingan fungsi keanggotaan *Gaussian* sebelum dan sesudah proses pelatihan (*training*).
  * **Inspeksi 5 Lapisan ANFIS:** Menampilkan nilai komputasi antara dari Lapisan 1 (fuzzifikasi) hingga Lapisan 5 (defuzzifikasi total).
  * **Evaluasi Regresi:** Mengukur performa model pada data uji menggunakan metrik $\text{RMSE}$, $\text{MAE}$, dan $R^2$.

## Struktur Direktori Proyek

```text
anfis-food-delivery/
├── data/
│   ├── .gitkeep
│   └── Food_Delivery_Times.csv     # Dataset utama dari Kaggle
├── src/
│   ├── __init__.py                 # Penanda paket modul src & ringkasan penanggung jawab
│   ├── preprocessing.py            # Pembersihan NaN, encoding, split 80:20, Min-Max scaling
│   ├── context_generator.py        # Generator kondisi operasional berbasis seed tanggal & jam
│   ├── lapis1.py                   # Lapisan 1: Fuzzifikasi Gaussian & Update Premis (GD)
│   ├── lapis2.py                   # Lapisan 2: Pembangkitan 32 Aturan & Firing Strength
│   ├── lapis3.py                   # Lapisan 3: Normalisasi Firing Strength
│   ├── lapis4.py                   # Lapisan 4: Defuzzifikasi Sugeno Orde-1 & Optimasi LSE
│   ├── lapis5.py                   # Lapisan 5: Agregasi Output Akhir & Perhitungan RMSE
│   └── anfis.py                    # Orkestrator utama yang menghubungkan Lapisan 1 - 5
├── app.py                          # Antarmuka grafis utama berbasis Streamlit
├── Dockerfile                      # Konfigurasi image Docker aplikasi
├── docker-compose.yml              # Orkestrasi container Docker
├── requirements.txt                # Daftar pustaka Python yang digunakan
├── .dockerignore                   # Pengecualian file saat build Docker
├── .gitignore                      # Pengecualian file saat push ke GitHub
└── README.md                       # Dokumentasi proyek dan panduan tim
```

## Panduan Instalasi & Menjalankan Aplikasi

Setiap anggota tim dapat menjalankan aplikasi melalui **Virtual Environment (`venv`)** untuk pengembangan kode sehari-hari, atau menggunakan **Docker** untuk memastikan lingkungan eksekusi yang seragam.

### Opsi 1: Menjalankan Secara Lokal (`venv`)

1. **Clone repository dari GitHub:**
   ```bash
   git clone https://github.com/USERNAME_ANDA/anfis-food-delivery.git
   cd anfis-food-delivery
   ```

2. **Buat dan aktifkan Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instal seluruh dependensi:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Jalankan server Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Akses aplikasi melalui browser di alamat `http://localhost:8501`.

### Opsi 2: Menjalankan Menggunakan Docker

Pastikan **Docker Desktop** (Windows/macOS) atau **Docker Engine** (Linux) sudah berjalan di perangkat Anda.

1. **Build dan jalankan container:**
   ```bash
   docker compose up --build -d
   ```

2. Buka browser dan akses `http://localhost:8501`.

3. **Menghentikan container setelah selesai digunakan:**
   ```bash
   docker compose down
   ```

## Panduan Penggunaan Ikon Antarmuka (Material Icons)

Untuk menjaga tampilan antarmuka tetap konsisten di berbagai sistem operasi (Windows, macOS, Linux, maupun Docker) dan terlihat profesional tanpa bergantung pada font emoji bawaan perangkat, seluruh komponen visual menggunakan **Google Material Symbols** yang sudah terintegrasi langsung di dalam Streamlit.

### Aturan Penulisan Ikon
* **Tanpa Library Tambahan:** Tidak perlu menginstal paket ikon pihak ketiga di `requirements.txt`.
* **Format Sintaks:** Gunakan pola `:material/nama_ikon:` dengan huruf kecil semua dan tanda garis bawah (`_`) sebagai pengganti spasi.
* **Katalog Nama Ikon:** Cari nama ikon yang diinginkan pada halaman resmi [Google Fonts Material Symbols](https://fonts.google.com/icons) (misalnya ikon *Local Shipping* ditulis menjadi `:material/local_shipping:`).

### Contoh Penerapan di Kode Streamlit

```python
# 1. Pada Judul, Subjudul, atau Tab
st.title(":material/delivery_dining: Estimasi Waktu Pengantaran")
tabs = st.tabs([":material/home: Beranda", ":material/analytics: Evaluasi"])

# 2. Pada Kartu Metrik (st.metric)
st.metric(":material/timer: Estimasi Waktu Tiba", "32.5 menit")

# 3. Pada Tombol & Kotak Notifikasi (parameter icon=)
st.button("Latih Model", icon=":material/play_arrow:")
st.info("Data berhasil dimuat.", icon=":material/info:")
st.warning("Model belum dilatih.", icon=":material/warning:")
```

## Panduan Kolaborasi Git & GitHub untuk Anggota Tim

Untuk mencegah konflik kode (*merge conflict*) saat 5 anggota bekerja secara bersamaan, ikuti aturan alur kerja berikut:

1. **Jangan melakukan push langsung ke branch `main`:** Branch `main` dijaga agar selalu dalam kondisi stabil dan siap didemonstrasikan.
2. **Selalu tarik pembaruan terbaru sebelum mulai bekerja:**
   ```bash
   git checkout main
   git pull origin main
   ```
3. **Buat branch baru sesuai lapisan atau fitur masing-masing:**
   ```bash
   git checkout -b nama-branch-anda
   # Contoh: git checkout -b fitur-lapis1 atau git checkout -b fitur-lapis4
   ```
4. **Simpan perubahan dan unggah ke branch Anda:**
   ```bash
   git add .
   git commit -m "Implementasi Lapisan X dan modul pendukung"
   git push -u origin nama-branch-anda
   ```
5. **Gabungkan kode melalui Pull Request (PR):**
   * Buka halaman repository di GitHub, klik **Compare & pull request** dari branch Anda menuju `main`.
   * Pastikan tidak ada konflik, lalu klik **Merge pull request** dan beri tahu anggota kelompok lain di grup diskusi agar mereka menjalankan `git pull origin main`.

## Panduan Menambah atau Memperbarui Library Python

Jika ada anggota tim yang membutuhkan *library* tambahan (misalnya `plotly` atau `seaborn` untuk visualisasi tambahan), **wajib** mengikuti langkah sinkronisasi berikut agar aplikasi tidak *error* di laptop anggota lain maupun di dalam Docker:

1. **Pastikan Virtual Environment aktif** (terdapat tanda `(venv)` di awal baris terminal).
2. **Instal *library* yang dibutuhkan:**
   ```bash
   pip install nama-library
   ```
3. **Daftarkan *library* tersebut ke file `requirements.txt`:**
   * Buka file `requirements.txt` dan tambahkan nama *library* baru pada baris paling bawah (contoh: menambahkan baris `plotly`).
   * **Catatan Penting:** Hindari menimpa seluruh isi `requirements.txt` menggunakan `pip freeze > requirements.txt` di Windows karena dapat memasukkan paket khusus sistem operasi Windows (seperti `pywin32`) yang akan menyebabkan proses *build* Docker (`python:3.11-slim` berbasis Linux) gagal.
4. **Uji ulang kompatibilitas Docker (opsional namun disarankan):**
   ```bash
   docker compose up --build -d
   ```
5. **Commit dan push perubahan `requirements.txt` ke GitHub** beserta kode baru Anda, lalu informasikan anggota tim lain untuk menjalankan perintah pembaruan di laptop masing-masing:
   ```bash
   git pull origin main
   pip install -r requirements.txt
   ```

## Pembagian Tugas & Kontribusi Anggota Kelompok

Setiap anggota bertanggung jawab penuh atas **1 lapisan inti ANFIS**, **modul pendukung/antarmuka terkait**, serta **bagian penulisan laporan dan validasi perhitungan manual** untuk lapisan tersebut:

| No. | Nama / NIM | Tanggung Jawab Lapisan ANFIS | Modul Pendukung & Antarmuka (`src/` & `app.py`) | Kontribusi Laporan & Validasi Manual |
| :--- | :--- | :--- | :--- | :--- |
| **1** | *Nama Anggota 1 (NIM)* | **Lapisan 1 (`src/lapis1.py`)**<br>Fuzzifikasi Gaussian ($\mu_{i,0}, \mu_{i,1}$) & *Gradient Descent* (*backward pass* untuk $c, \sigma$). | `src/preprocessing.py` (*cleaning* `NaN`, *ordinal encoding*, *Min-Max scaling*), Tab 2 GUI, dan grafik pergeseran kurva MF (*Before vs After*). | **BAB I (Pendahuluan)**, **BAB II (Konsep Fuzzifikasi & MF Gaussian)**, serta contoh hitung manual **Lapisan 1**. |
| **2** | *Nama Anggota 2 (NIM)* | **Lapisan 2 (`src/lapis2.py`)**<br>Pembangkitan 32 kombinasi aturan (`itertools.product`) & *Firing Strength* ($w_r$). | `src/context_generator.py` (*seed* tanggal & jam) serta kartu kondisi operasional *live* pada Tab 1 GUI. | **BAB II (Basis Aturan Sugeno & Operator AND)**, **BAB III (Definisi Variabel & Mekanisme Seed)**, serta contoh hitung manual **Lapisan 2**. |
| **3** | *Nama Anggota 3 (NIM)* | **Lapisan 3 (`src/lapis3.py`)**<br>Normalisasi *Firing Strength* ($\bar{w}_r = w_r / \sum w_k$) & integrasi `src/anfis.py`. | Kerangka utama `app.py`, manajemen `st.session_state`, alur navigasi, konfigurasi Docker, `README.md`, dan Tab 6 GUI. | **BAB III (Arsitektur Sistem 5 Lapisan, Flowchart & Desain Antarmuka)**, Daftar Pustaka, serta contoh hitung manual **Lapisan 3**. |
| **4** | *Nama Anggota 4 (NIM)* | **Lapisan 4 (`src/lapis4.py`)**<br>Defuzzifikasi fungsi linear Sugeno orde-1 ($f_r, \bar{w}_r f_r$) & optimasi *Least Squares Estimator* (LSE). | Tab 3 GUI (Konfigurasi *Training* & Grafik Konvergensi RMSE) serta Tab 4 GUI (Simulasi Input & Tabel Inspeksi 5 Lapisan). | **BAB II (Algoritma Hybrid Learning: LSE + GD)**, **BAB IV (Subbab 4.1–4.5: Implementasi Kode & GUI)**, serta contoh hitung manual **Lapisan 4**. |
| **5** | *Nama Anggota 5 (NIM)* | **Lapisan 5 (`src/lapis5.py`)**<br>Agregasi output akhir ($\hat{y} = \sum \bar{w}_r f_r$) & perhitungan galat pelatihan ($\text{RMSE}$). | Fungsi denormalisasi menit, Tab 5 GUI (Metrik $\text{RMSE}$, $\text{MAE}$, $R^2$ & Grafik Aktual vs Prediksi), serta validasi tabel Excel. | **BAB IV (Subbab 4.6–4.8: Hasil Pengujian, Evaluasi & Analisis)**, **BAB V (Penutup)**, kompilasi akhir laporan & slide presentasi, serta hitung manual **Lapisan 5**. |

## Urutan Pengerjaan Proyek (*Roadmap* 4 Tahap)

Karena kontrak input/output tiap file sudah dikunci di dalam catatan *docstring*, kelima anggota dapat bekerja secara paralel sesuai tahapan berikut:

1. **Tahap 1 — Fondasi Data, Kerangka Utama & Penulisan BAB I–II (Hari 1–2):**
   * **Anggota 3** menyiapkan repository GitHub, struktur folder lengkap, kerangka `app.py`, dan `src/anfis.py`, lalu mengundang seluruh anggota sebagai *Collaborator*.
   * **Anggota 1** mengunduh dataset `Food_Delivery_Times.csv` dari Kaggle ke folder `data/`, menyelesaikan `src/preprocessing.py` dan `src/lapis1.py`, serta mulai menyusun **BAB I**.
   * **Anggota 2** menyelesaikan `src/context_generator.py` dan `src/lapis2.py`, serta mulai menyusun bagian landasan aturan fuzzy di **BAB II**.
2. **Tahap 2 — Penyelesaian Lapisan 3–5 & Aktivasi Mesin ANFIS Penuh (Hari 3–5):**
   * **Anggota 3** mengimplementasikan normalisasi pada `src/lapis3.py`.
   * **Anggota 4** mengimplementasikan persamaan linear Sugeno dan matriks LSE pada `src/lapis4.py`, serta menulis teori *Hybrid Learning* di **BAB II**.
   * **Anggota 5** mengimplementasikan penjumlahan output akhir dan perhitungan $\text{RMSE}$ pada `src/lapis5.py`.
   * Seluruh anggota melakukan *Pull Request* ke branch `main`. Begitu `lapis1.py` s.d. `lapis5.py` terisi, `app.py` otomatis beralih dari *Stub Mode* ke mesin ANFIS asli.
3. **Tahap 3 — Penyempurnaan Antarmuka, Validasi Manual Estafet & BAB III–IV (Hari 6–8):**
   * Setiap anggota memverifikasi tampilan tab GUI bagiannya masing-masing (Tab 1 hingga Tab 6) dan mengambil tangkapan layar (*screenshot*) beresolusi jelas untuk laporan.
   * **Anggota 1 sampai Anggota 5** secara berurutan (estafet) menghitung dan mencocokkan angka hasil **Lapisan 1 $\rightarrow$ Lapisan 5** menggunakan 1 sampel data uji dari Tab 4 ke dalam tabel Excel/laporan untuk membuktikan kesesuaian hitungan manual dengan program Python.
   * **Anggota 3 & Anggota 4** menyelesaikan penulisan **BAB III** dan **BAB IV (Implementasi)**.
4. **Tahap 4 — Evaluasi Akhir, Uji Docker, BAB V & Gladi Presentasi (Hari 9–10):**
   * **Anggota 5** mencatat skor akhir $\text{RMSE}$, $\text{MAE}$, dan $R^2$ pada Tab 5, melengkapi **BAB IV (Analisis)** dan **BAB V (Penutup)**, serta menyusun *slide* presentasi.
   * **Anggota 3** mengompilasi seluruh bab menjadi dokumen laporan final dan menguji jalannya aplikasi di dalam container Docker (`docker compose up --build -d`).
   * Seluruh tim melakukan latihan demonstrasi langsung aplikasi (mode *Live Seed* dan *Input Manual*) dan memastikan setiap anggota siap menjelaskan kode serta rumus pada lapisannya masing-masing.