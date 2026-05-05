
# ============================================================
# dashboard.py
# Project : Chatbot Edukatif IPA SD — CC26-PSU312
# Streamlit Dashboard
# Run     : streamlit run dashboard/dashboard.py
# ============================================================

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# 0. CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard IPA SD — CC26-PSU312",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .metric-card {
            background: #F7F9FC;
            border: 1px solid #E3E8F2;
            border-radius: 16px;
            padding: 18px 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .metric-val {
            font-size: 28px;
            font-weight: 800;
            color: #163A6B;
            line-height: 1.1;
        }
        .metric-lbl {
            font-size: 12px;
            color: #5C677D;
            margin-top: 6px;
        }
        .soft-box {
            background: #F8FBFF;
            border: 1px solid #E3ECF7;
            border-radius: 14px;
            padding: 14px 16px;
        }
        .warn-box {
            background: #FFF8E8;
            border-left: 4px solid #A86A00;
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 13px;
            color: #000000;
        }
        .warn-box b {
            color: #000000;
        }
        .ok-box {
            background: #EAF7EE;
            border-left: 4px solid #177245;
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = {
    "primary": "#2F6FDB",
    "secondary": "#2E8B57",
    "accent": "#D18A00",
    "danger": "#C23B22",
    "neutral": "#65758B",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFD",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E8ECF3",
        "grid.linewidth": 0.6,
        "font.size": 9,
    }
)

# ────────────────────────────────────────────────────────────
# 1. DATA LOADER
# ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "data_cleran" / "datasoal_clean.csv",
        base_dir.parent / "data" / "datasoal.csv",
        base_dir.parent / "data_clean.csv",
        base_dir / "data_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "Dataset tidak ditemukan. Letakkan file CSV di dashboard/data_cleran/datasoal_clean.csv "
            "atau data/datasoal.csv."
        )

    # Standardize text columns
    for col in ["topik", "subtopik", "soal", "jawaban", "contoh", "konteks"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "topik" in df.columns:
        df["topik"] = df["topik"].str.lower()
    if "subtopik" in df.columns:
        df["subtopik"] = df["subtopik"].str.lower()

    if "link_sumber" not in df.columns and "link sumber buku" in df.columns:
        df = df.rename(columns={"link sumber buku": "link_sumber"})

    if "kelas" not in df.columns:
        df["kelas"] = 5

    if "soal_len" not in df.columns and "soal" in df.columns:
        df["soal_len"] = df["soal"].str.len()
    if "jawaban_len" not in df.columns and "jawaban" in df.columns:
        df["jawaban_len"] = df["jawaban"].str.len()

    if "kata_awal" not in df.columns and "soal" in df.columns:
        df["kata_awal"] = df["soal"].apply(
            lambda s: str(s).strip().split()[0].lower().rstrip("!?,.") if str(s).strip() else "-"
        )

    return df


df = load_data()

# ────────────────────────────────────────────────────────────
# 2. SIDEBAR FILTER
# ────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/128/2784/2784403.png", width=64)
st.sidebar.title("🔬 Dashboard IPA SD")
st.sidebar.caption("CC26-PSU312 — Coding Camp 2026")
st.sidebar.divider()

all_topik = sorted(df["topik"].dropna().unique().tolist())
sel_topik = st.sidebar.multiselect("Filter topik", all_topik, default=all_topik)

min_soal = int(df["soal_len"].min())
max_soal = int(df["soal_len"].max())
range_soal = st.sidebar.slider(
    "Panjang soal (karakter)",
    min_value=min_soal,
    max_value=max_soal,
    value=(min_soal, max_soal),
)

st.sidebar.divider()
st.sidebar.caption("Data: datasoal.csv\nVersi bersih: datasoal_clean.csv")

df_fil = df[
    df["topik"].isin(sel_topik) & df["soal_len"].between(range_soal[0], range_soal[1])
].copy()

# ────────────────────────────────────────────────────────────
# 3. HEADER
# ────────────────────────────────────────────────────────────
st.title("📊 Dashboard Analisis Dataset Chatbot IPA SD")
st.write(
    "Insight dari **datasoal.csv** — pasangan soal, jawaban, contoh, dan konteks untuk IPA kelas 5."
)
st.divider()

# ────────────────────────────────────────────────────────────
# 4. METRICS
# ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5, gap="small")

metrics = [
    (c1, len(df_fil), "Pasangan Soal-Jawaban"),
    (c2, df_fil["topik"].nunique(), "Topik unik"),
    (c3, df_fil["subtopik"].nunique(), "Subtopik unik"),
    (c4, f"{df_fil['soal_len'].mean():.0f}", "Rata-rata panjang soal"),
    (c5, f"{df_fil['jawaban_len'].mean():.0f}", "Rata-rata panjang jawaban"),
]

for col, val, lbl in metrics:
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

def show_fig(fig):
    """Render figure safely so titles/labels are not clipped."""
    fig.set_constrained_layout(True)
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)

# ────────────────────────────────────────────────────────────
# 5. TABS
# ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Distribusi Topik", "📏 Panjang Teks", "🔍 Subtopik", "💡 Pola Pertanyaan", "✅ Kesiapan Data"]
)

# TAB 1
with tab1:
    st.subheader("BQ1 — Topik IPA mana yang paling banyak tersedia datanya?")

    topik_counts = df_fil["topik"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(topik_counts) * 0.38)))
    colors = [
        PALETTE["danger"] if c < 20 else PALETTE["accent"] if c < 50 else PALETTE["primary"]
        for c in topik_counts.values
    ]
    bars = ax.barh(topik_counts.index, topik_counts.values, color=colors, height=0.72, edgecolor="white")

    for bar, val in zip(bars, topik_counts.values):
        ax.text(val + max(topik_counts.max() * 0.01, 0.25), bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8)

    ax.axvline(20, color=PALETTE["danger"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(50, color=PALETTE["accent"], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Jumlah pasangan soal-jawaban")
    ax.set_ylabel("Topik")
    ax.set_title("Distribusi Data per Topik", fontweight="bold", pad=12)
    ax.margins(x=0.15)

    legend_items = [
        mpatches.Patch(color=PALETTE["danger"], label="< 20 (perlu augmentasi)"),
        mpatches.Patch(color=PALETTE["accent"], label="20–50 (cukup)"),
        mpatches.Patch(color=PALETTE["primary"], label="> 50 (baik)"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="lower right")

    show_fig(fig)

    needs_aug = topik_counts[topik_counts < 20]
    if len(needs_aug) > 0:
        st.markdown(
            f'<div class="warn-box">⚠️ <b>{len(needs_aug)} topik</b> masih di bawah 20 data: '
            + ", ".join([f"<b>{t}</b> ({v})" for t, v in needs_aug.items()])
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="ok-box">✅ Semua topik sudah memiliki ≥ 20 data.</div>', unsafe_allow_html=True)

# TAB 2
with tab2:
    st.subheader("BQ2 — Bagaimana distribusi panjang soal dan jawaban?")

    col_a, col_b = st.columns(2, gap="large")

    for col, field, color, label in [
        (col_a, "soal_len", PALETTE["primary"], "Soal"),
        (col_b, "jawaban_len", PALETTE["secondary"], "Jawaban"),
    ]:
        with col:
            data = df_fil[field].dropna()
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.hist(data, bins=24, color=color, alpha=0.9, edgecolor="white")
            ax.axvline(data.mean(), color=PALETTE["danger"], linestyle="--", linewidth=1.2, label=f"Mean = {data.mean():.0f}")
            ax.axvline(data.median(), color=PALETTE["accent"], linestyle=":", linewidth=1.4, label=f"Median = {data.median():.0f}")
            ax.set_xlabel("Karakter")
            ax.set_ylabel("Frekuensi")
            ax.set_title(f"Distribusi Panjang {label}", fontweight="bold", pad=12)
            ax.legend(fontsize=8, loc="upper right")
            ax.margins(y=0.08)
            show_fig(fig)
            st.caption(
                f"Min: {int(data.min())} | Q1: {data.quantile(0.25):.0f} | "
                f"Median: {data.median():.0f} | Q3: {data.quantile(0.75):.0f} | Max: {int(data.max())}"
            )

    st.markdown("##### Korelasi panjang soal dengan panjang jawaban")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = df_fil["soal_len"].astype(float)
    y = df_fil["jawaban_len"].astype(float)
    ax.scatter(x, y, alpha=0.25, s=14, color=PALETTE["primary"], edgecolors="none")

    if len(df_fil) >= 2:
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color=PALETTE["danger"], linewidth=1.6, label="Tren linear")
        corr = df_fil[["soal_len", "jawaban_len"]].corr().iloc[0, 1]
    else:
        corr = np.nan

    ax.set_xlabel("Panjang soal (karakter)")
    ax.set_ylabel("Panjang jawaban (karakter)")
    ax.set_title(f"Scatter: Panjang Soal vs Jawaban (r = {corr:.2f})", fontweight="bold", pad=12)
    ax.legend(fontsize=8)
    ax.margins(0.08)
    show_fig(fig)

    st.info(
        f"Korelasi r = **{corr:.2f}** — "
        + ("ada kecenderungan jawaban lebih panjang saat soal lebih panjang." if corr > 0.3 else "hubungannya tidak terlalu kuat.")
    )

# TAB 3
with tab3:
    st.subheader("BQ3 — Subtopik apa yang paling banyak muncul?")

    sel_t = st.selectbox("Pilih topik untuk drill-down:", options=sorted(df_fil["topik"].unique()))
    sub_df = df_fil[df_fil["topik"] == sel_t]["subtopik"].value_counts().head(15).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(5, len(sub_df) * 0.42)))
    ax.barh(sub_df.index, sub_df.values, color=PALETTE["primary"], height=0.7, edgecolor="white")
    for i, (idx, val) in enumerate(sub_df.items()):
        ax.text(val + max(sub_df.max() * 0.01, 0.2), i, str(val), va="center", fontsize=8.5)

    ax.set_xlabel("Jumlah soal")
    ax.set_title(f"Top Subtopik — {sel_t.title()}", fontweight="bold", pad=12)
    ax.set_xlim(0, sub_df.max() * 1.2 if len(sub_df) else 1)
    ax.margins(x=0.12)
    show_fig(fig)

    st.divider()
    st.markdown("##### Contoh soal-jawaban")
    sel_sub = st.selectbox("Pilih subtopik:", options=sorted(df_fil[df_fil["topik"] == sel_t]["subtopik"].unique()))
    sample_rows = df_fil[(df_fil["topik"] == sel_t) & (df_fil["subtopik"] == sel_sub)][["soal", "jawaban", "konteks"]].head(5)

    for _, row in sample_rows.iterrows():
        with st.expander(f"❓ {row['soal']}"):
            st.markdown(f"**Jawaban:** {row['jawaban']}")
            st.markdown(f"*Konteks: {row['konteks']}*")

# TAB 4
with tab4:
    st.subheader("BQ4 — Seberapa beragam variasi pertanyaan?")

    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        kata_cnt = df_fil["kata_awal"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(kata_cnt.index, kata_cnt.values, color=PALETTE["primary"], edgecolor="white")
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(kata_cnt.max() * 0.01, 0.3),
                str(int(bar.get_height())),
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel("Kata pembuka soal")
        ax.set_ylabel("Frekuensi")
        ax.set_title("Pola Kata Pembuka Soal", fontweight="bold", pad=12)
        plt.xticks(rotation=35, ha="right")
        show_fig(fig)

    with col_b:
        st.markdown('<div class="soft-box">', unsafe_allow_html=True)
        total = len(df_fil)
        for kata, cnt in kata_cnt.head(6).items():
            pct = (cnt / total * 100) if total else 0
            st.write(f"**{kata}** — {cnt} soal ({pct:.1f}%)")
        st.caption("Variasi kata pembuka membantu chatbot mengenali bentuk pertanyaan yang berbeda.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Cari soal berdasarkan kata kunci")
    keyword = st.text_input("Masukkan kata kunci", placeholder="contoh: fotosintesis")
    if keyword:
        result = df_fil[
            df_fil["soal"].str.contains(keyword, case=False, na=False)
            | df_fil["jawaban"].str.contains(keyword, case=False, na=False)
        ]
        st.info(f"Ditemukan **{len(result)}** pasangan soal-jawaban yang mengandung '{keyword}'.")
        for _, row in result.head(8).iterrows():
            with st.expander(f"❓ {row['soal'][:90]}"):
                st.write(f"**Topik:** {row['topik']} → {row['subtopik']}")
                st.write(f"**Jawaban:** {row['jawaban']}")
                st.write(f"**Konteks:** {row['konteks']}")

# TAB 5
with tab5:
    st.subheader("BQ5 — Apakah data sudah merata dan siap untuk training chatbot?")

    topik_stats = df_fil.groupby("topik").agg(
        jumlah_soal=("soal", "count"),
        subtopik_unik=("subtopik", "nunique"),
        rata_jawab=("jawaban_len", "mean"),
    ).sort_values("jumlah_soal", ascending=False)

    if len(topik_stats) > 0:
        topik_stats["skor"] = (
            (topik_stats["jumlah_soal"] / topik_stats["jumlah_soal"].max()) * 40
            + (topik_stats["subtopik_unik"] / topik_stats["subtopik_unik"].max()) * 30
            + (topik_stats["rata_jawab"] / topik_stats["rata_jawab"].max()) * 30
        ).round(1)
    else:
        topik_stats["skor"] = []

    topik_stats["status"] = topik_stats["skor"].apply(
        lambda s: "🟢 Siap" if s >= 60 else ("🟡 Cukup" if s >= 40 else "🔴 Perlu augmentasi")
    )

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        skor = topik_stats["skor"].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(11, max(6, len(skor) * 0.42)))
        bar_colors = [
            PALETTE["danger"] if s < 40 else PALETTE["accent"] if s < 60 else PALETTE["secondary"]
            for s in skor.values
        ]
        ax.barh(skor.index, skor.values, color=bar_colors, height=0.7, edgecolor="white")
        ax.axvline(40, color=PALETTE["danger"], linestyle="--", linewidth=1, alpha=0.7, label="Min 40")
        ax.axvline(60, color=PALETTE["secondary"], linestyle="--", linewidth=1, alpha=0.7, label="Target 60")
        for i, (idx, val) in enumerate(skor.items()):
            ax.text(val + 0.6, i, f"{val:.0f}", va="center", fontsize=8.5)
        ax.set_xlabel("Skor kesiapan data (0–100)")
        ax.set_title("Skor Kesiapan per Topik", fontweight="bold", pad=12)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 110)
        ax.margins(x=0.1)
        show_fig(fig)

    with col_b:
        st.markdown("##### Status per topik")
        display = topik_stats[["jumlah_soal", "subtopik_unik", "skor", "status"]].copy()
        display.columns = ["Soal", "Subtopik", "Skor", "Status"]
        st.dataframe(display, use_container_width=True, height=390)

    st.divider()

    siap = int((topik_stats["skor"] >= 60).sum())
    cukup = int(((topik_stats["skor"] >= 40) & (topik_stats["skor"] < 60)).sum())
    perlu_aug = int((topik_stats["skor"] < 40).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("🟢 Siap Training", f"{siap} topik")
    m2.metric("🟡 Cukup", f"{cukup} topik")
    m3.metric("🔴 Perlu Augmentasi", f"{perlu_aug} topik")

    st.markdown(
        """
        **Catatan metodologi skor:**
        - 40% bobot jumlah soal
        - 30% bobot jumlah subtopik unik
        - 30% bobot rata-rata panjang jawaban
        """
    )

    st.divider()
    st.subheader("📋 Eksplorasi data")
    n_sample = st.slider("Tampilkan n baris", 5, 50, 10)
    st.dataframe(
        df_fil[["topik", "subtopik", "soal", "jawaban", "konteks"]].head(n_sample),
        use_container_width=True,
    )

# ────────────────────────────────────────────────────────────
# 6. FOOTER
# ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📌 CC26-PSU312 · Coding Camp 2026 powered by DBS Foundation | "
    "Dataset: datasoal.csv | Tech Stack: Python · Pandas · Matplotlib · Streamlit"
)
