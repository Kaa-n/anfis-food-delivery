"""
Paket modul antarmuka pengguna (UI) per tab untuk aplikasi ANFIS Food Delivery.

Struktur Pembagian Tab:
- components.py     : Komponen UI bersama, styling CSS Streamlit, dan fallback/stub engine
- tab1_beranda.py   : Tab 1 - Beranda, Live Seed Kondisi Operasional & Prediksi Cepat
- tab2_dataset.py   : Tab 2 - Manajemen Dataset CSV, Preprocessing & Pratinjau Data
- tab3_training.py  : Tab 3 - Konfigurasi Hybrid Learning, Grafik Loss & Kurva Gaussian
- tab4_simulasi.py  : Tab 4 - Simulasi Skenario & Inspeksi Perhitungan Lapisan 1-5
- tab5_evaluasi.py  : Tab 5 - Evaluasi Performa Model (RMSE, MAE, R2) & Grafik Aktual vs Prediksi
- tab6_informasi.py : Tab 6 - Panduan Sistem, Parameter Premis & Matriks 32 Aturan Sugeno
"""

from tabs.tab1_beranda import render_tab1_beranda
from tabs.tab2_dataset import render_tab2_dataset
from tabs.tab3_training import render_tab3_training
from tabs.tab4_simulasi import render_tab4_simulasi
from tabs.tab5_evaluasi import render_tab5_evaluasi
from tabs.tab6_informasi import render_tab6_informasi

__all__ = [
    "render_tab1_beranda",
    "render_tab2_dataset",
    "render_tab3_training",
    "render_tab4_simulasi",
    "render_tab5_evaluasi",
    "render_tab6_informasi",
]