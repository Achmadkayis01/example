import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re

# -------------------------- CONFIG & HEADER --------------------------
st.set_page_config(page_title="Prediksi Harga Laptop", layout="wide")
st.title("💻📈 Prediksi Harga Laptop Berdasarkan Spesifikasi Menggunakan Regresi Linear")

# -------------------------- LOAD & PREPARE DATA --------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.head(10)

    # Kolom harga → int
    df.rename(columns={"latest_price": "Harga"}, inplace=True)
    df["Harga"] = pd.to_numeric(df["Harga"], errors="coerce").fillna(0).astype(int)

    # Ekstrak angka dari kolom RAM / SSD / HDD
    def _extract_num(series, default=0):
        return (
            series.astype(str)
            .str.extract(r"(\d+)")
            .fillna(default)
            .astype(int)
            .squeeze()
        )

    df["RAM_GB"] = _extract_num(df["ram_gb"])
    df["SSD_GB"] = _extract_num(df["ssd"])
    df["HDD_GB"] = _extract_num(df["hdd"])

    # Bersihkan kolom yang akan ditampilkan
    show_cols = [
        "brand",
        "model",
        "processor_brand",
        "processor_gnrtn",
        "RAM_GB",
        "SSD_GB",
        "HDD_GB",
        "graphic_card_gb",
        "os",
        "Harga",
        "star_rating",
    ]
    df = df[show_cols].copy()

    return df


DATA_PATH = "Cleaned_Laptop_data.csv"
df = load_data(DATA_PATH)

# -------------------------- SIDEBAR FILTERS --------------------------
st.sidebar.header("⚙️ Filter Spesifikasi")

# Brand filter
brands = ["Semua"] + sorted(df["brand"].unique().tolist())
brand_choice = st.sidebar.selectbox("Brand", brands)
filtered_df = df if brand_choice == "Semua" else df[df["brand"] == brand_choice]

# Range sliders
ram_min, ram_max = int(df["RAM_GB"].min()), int(df["RAM_GB"].max())
ssd_min, ssd_max = int(df["SSD_GB"].min()), int(df["SSD_GB"].max())
hdd_min, hdd_max = int(df["HDD_GB"].min()), int(df["HDD_GB"].max())

ram_sel = st.sidebar.slider("RAM (GB)", ram_min, ram_max, (ram_min, ram_max))
ssd_sel = st.sidebar.slider("SSD (GB)", ssd_min, ssd_max, (ssd_min, ssd_max))
hdd_sel = st.sidebar.slider("HDD (GB)", hdd_min, hdd_max, (hdd_min, hdd_max))

# Apply numeric filters
filtered_df = filtered_df[
    (filtered_df["RAM_GB"].between(*ram_sel))
    & (filtered_df["SSD_GB"].between(*ssd_sel))
    & (filtered_df["HDD_GB"].between(*hdd_sel))
]

st.subheader("📋 Data Laptop (setelah filter)")
st.dataframe(filtered_df, use_container_width=True)

# -------------------------- LINEAR REGRESSION MODEL --------------------------
# Fitur numerik sederhana
X = df[["RAM_GB", "SSD_GB", "HDD_GB"]]
y = df["Harga"]
model = LinearRegression()
model.fit(X, y)

st.markdown("---")
st.header("🔮 Prediksi Harga")

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Pilih Spesifikasi untuk Diprediksi")
    input_ram = st.slider("RAM (GB)", ram_min, ram_max, 8, step=2)
    input_ssd = st.slider("SSD (GB)", ssd_min, ssd_max, 256, step=128)
    input_hdd = st.slider("HDD (GB)", hdd_min, hdd_max, 0, step=500)

    if st.button("Prediksi Harga Laptop"):
        pred_price = model.predict([[input_ram, input_ssd, input_hdd]])[0]
        st.success(f"Perkiraan harga laptop dengan spesifikasi tersebut: **Rp {int(pred_price):,}**")

with col_r:
    st.markdown("#### Korelasi Fitur vs Harga")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["RAM_GB"], df["Harga"], label="RAM", alpha=0.6, s=60)
    ax.set_xlabel("RAM (GB)")
    ax.set_ylabel("Harga (Rp)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("Rp {x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

# -------------------------- SHOPPING CART SECTION --------------------------
st.markdown("---")
st.header("🛒 Hitung Total Harga Pembelian")

st.markdown(
    "Masukkan jumlah unit (Quantity) yang ingin dibeli untuk setiap laptop di tabel berikut:"
)
qty_inputs, totals = [], []
for idx, row in filtered_df.iterrows():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"**{row['brand']} {row['model']}** | {row['processor_brand']} Gen {row['processor_gnrtn']} | "
            f"{row['RAM_GB']} GB RAM / {row['SSD_GB']} GB SSD / {row['HDD_GB']} GB HDD ‑ "
            f"⭐ {row['star_rating']}"
        )
    with col2:
        qty = st.number_input(
            "Qty",
            min_value=0,
            step=1,
            key=f"qty_{idx}",
            label_visibility="collapsed",
        )
    qty_inputs.append(qty)
    totals.append(qty * row["Harga"])

filtered_df = filtered_df.copy()
filtered_df["Quantity"] = qty_inputs
filtered_df["Total_Harga"] = totals
purchase_df = filtered_df[filtered_df["Quantity"] > 0]

if not purchase_df.empty:
    st.subheader("💰 Ringkasan Pembelian")
    tmp = purchase_df[
        [
            "brand",
            "model",
            "RAM_GB",
            "SSD_GB",
            "HDD_GB",
            "Harga",
            "Quantity",
            "Total_Harga",
        ]
    ].copy()
    tmp["Harga"] = tmp["Harga"].apply(lambda x: f"Rp {x:,.0f}")
    tmp["Total_Harga"] = tmp["Total_Harga"].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(tmp, use_container_width=True)

    grand_total = purchase_df["Total_Harga"].sum()
    st.markdown(f"### 🏷️ **TOTAL KESELURUHAN: Rp {grand_total:,.0f}**")
else:
    st.info("Pilih minimal satu laptop dan quantity > 0 untuk menampilkan ringkasan.")

# -------------------------- FOOTER --------------------------
st.caption(
    "Dataset: Cleaned_Laptop_data.csv • Model: LinearRegression (fit on RAM/SSD/HDD) • © 2025"
)
