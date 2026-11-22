"""Ekspor contoh estafet dari sampel pertama data uji, tanpa mengubah dataset."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from src.anfis import ANFIS
from src.preprocessing import preprocess_data

dataset = Path("data/Food_Delivery_Times.csv")
_, X_train, X_test, y_train, _, scaler = preprocess_data(
    pd.read_csv(dataset), test_size=0.2, random_state=42)
model = ANFIS(lr=0.05, reg=0.001)
model.fit(X_train, y_train, epochs=10)
x = X_test[:1]
_, _, weights, f, o4, prediction = model.forward_layers(x)
table = pd.DataFrame({"Rule": [f"R{i+1}" for i in range(32)],
                      "w_bar": weights[0]})
for i in range(5):
    table[f"x{i+1}"] = x[0, i]
    table[f"p{i+1}"] = model.consequents[:, i]
    table[f"p{i+1}_x{i+1}"] = model.consequents[:, i] * x[0, i]
table["q"] = model.consequents[:, 5]
table["f"] = f[0]
table["o4"] = o4[0]
table.to_csv("docs/hitung_manual_lapis4.csv", index=False, encoding="utf-8-sig")
r = int(np.argmax(weights[0]))
products = model.consequents[r, :5] * x[0]
manual_f = sum(float(v) for v in products) + float(model.consequents[r, 5])
manual_o4 = float(weights[0, r]) * manual_f
np.testing.assert_allclose(manual_o4, o4[0, r], atol=1e-12)
text = f"""# Hitung Manual Lapisan 4 — Sampel Data Uji Proyek

Sumber: data/Food_Delivery_Times.csv.
SHA-256 dataset: {hashlib.sha256(dataset.read_bytes()).hexdigest()}.
Split test_size=0.2, random_state=42; sampel X_test pertama.
Pelatihan: 10 epoch, learning rate 0.05, lambda 0.001.
Data latih: {len(X_train)}; data uji: {len(X_test)}.
Angka ini berasal dari implementasi ANFIS asli, bukan mode fallback.

Input ternormalisasi: {x[0].tolist()}.

Aturan yang dijabarkan: R{r+1}, kombinasi MF {model.rules[r]}.
Koefisien p1–p5 dan q: {model.consequents[r].tolist()}.

Perkalian p_i*x_i:
{products.tolist()}.

f_R{r+1} = {" + ".join(f"({v:.12f})" for v in products)}
+ ({model.consequents[r, 5]:.12f}) = {manual_f:.12f}.

O4_R{r+1} = {weights[0, r]:.12f} × {manual_f:.12f}
= {manual_o4:.12f}.

Selisih perhitungan skalar dengan output program: {abs(manual_o4-o4[0,r]):.3e}.

Seluruh 32 aturan tersedia pada [CSV hitung manual](hitung_manual_lapis4.csv).
Jumlah kontribusi untuk diserahkan ke Anggota 5: {prediction[0]:.12f}.
Rentang target scaler: {scaler["y_min"]}–{scaler["y_max"]} menit.

Catatan: preprocessing proyek saat ini menghitung rentang Min-Max sebelum split.
Ini mengikuti kode anggota terkait; evaluasi akhir perlu meninjau potensi
kebocoran informasi rentang data uji. Dokumen ini membuktikan aritmetika Lapisan 4,
bukan menyimpulkan kualitas generalisasi model.
"""
Path("docs/hitung_manual_lapis4.md").write_text(text, encoding="utf-8")
print(f"R{r+1}: f={manual_f:.12f}; w_bar={weights[0,r]:.12f}; o4={manual_o4:.12f}")
print("CSV dan penjabaran manual tersimpan di docs/.")
