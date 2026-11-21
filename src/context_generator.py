
"""
================================================================================
MODUL GENERATOR KONTEKS OPERASIONAL (SEED TANGGAL & JAM)
Penanggung Jawab : Anggota 2
================================================================================

Menghasilkan simulasi kondisi operasional pengantaran makanan
secara deterministik berdasarkan tanggal dan jam.

Fungsi:
get_current_operational_context(custom_dt=None)

Output:
Dictionary berisi:
- timestamp_str
- seed
- Weather
- Traffic_Level
- Preparation_Time_min
- Courier_Experience_yrs
================================================================================
"""

from datetime import datetime
import numpy as np


def get_current_operational_context(custom_dt=None):
    """
    Menghasilkan kondisi operasional berdasarkan seed tanggal dan jam.

    Parameters
    ----------
    custom_dt : datetime, optional
        Waktu simulasi yang dapat ditentukan secara manual.
        Jika None, menggunakan waktu sekarang.

    Returns
    -------
    dict
        Informasi kondisi operasional.
    """

    # 1. Mengambil waktu referensi
    now = custom_dt if custom_dt is not None else datetime.now()

    if not isinstance(now, datetime):
        raise TypeError("custom_dt harus berupa objek datetime atau None.")

    # 2. Membuat seed berdasarkan tahun, bulan, tanggal, dan jam
    seed_value = int(now.strftime("%Y%m%d%H"))

    # Generator acak deterministik
    rng = np.random.default_rng(seed_value)

    hour = now.hour

    # 3. Mengatur probabilitas berdasarkan waktu operasional
    # Jam sibuk makan siang dan makan malam
    if 11 <= hour <= 14 or 17 <= hour <= 20:
        traffic_probs = [0.15, 0.45, 0.40]
        prep_base = (15.0, 28.0)

    # Jam operasional normal
    elif 6 <= hour <= 22:
        traffic_probs = [0.45, 0.40, 0.15]
        prep_base = (8.0, 18.0)

    # Jam dini hari
    else:
        traffic_probs = [0.75, 0.20, 0.05]
        prep_base = (5.0, 12.0)

    traffic_level = rng.choice(
        ["Low", "Medium", "High"],
        p=traffic_probs
    )

    weather = rng.choice(
        ["Clear", "Windy", "Foggy", "Rainy", "Snowy"],
        p=[0.50, 0.20, 0.10, 0.18, 0.02]
    )

    preparation_time = rng.uniform(
        prep_base[0], prep_base[1]
    )

    courier_experience = rng.uniform(1.0, 7.5)


    # 5. Format tanggal dalam Bahasa Indonesia
    nama_bulan = [
        "Januari", "Februari", "Maret", "April",
        "Mei", "Juni", "Juli", "Agustus",
        "September", "Oktober", "November", "Desember"
    ]

    timestamp_str = (
        f"{now.day} {nama_bulan[now.month - 1]} {now.year}, "
        f"Pukul {now.strftime('%H:%M')}"
    )

    # 6. Mengembalikan dictionary sesuai kontrak app.py
    context = {
        "timestamp_str": timestamp_str,
        "seed": seed_value,
        "Weather": str(weather),
        "Traffic_Level": str(traffic_level),
        "Preparation_Time_min": round(float(preparation_time), 2),
        "Courier_Experience_yrs": round(float(courier_experience), 2)
    }

    return context
