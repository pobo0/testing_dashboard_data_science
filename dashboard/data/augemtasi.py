# ============================================================
# 04_augmentasi.py
# Proyek  : Chatbot Edukatif IPA SD — CC26-PSU312
# Tahap   : Data Augmentation — Generate Q&A baru via Claude API
#
# Strategi augmentasi:
#   1. Topic Merging  — normalisasi topik-topik serupa ke topik induk
#   2. Rule-based     — variasi kata tanya, aktif↔pasif, sinonim,
#                       reformulasi kalimat, variasi konteks & contoh
#   3. AI-based       — Claude API generate pasangan soal-jawaban baru
#      (digunakan untuk topik dengan < 20 baris yang sangat kritis)
#
# Jalankan: python data/04_augmentasi.py
# Output  : data_clean/datasoal_augmented.csv
# ============================================================

import pandas as pd
import numpy as np
import re
import json
import time
import os
import random
import urllib.request
import urllib.error
from pathlib import Path
from itertools import cycle

# ────────────────────────────────────────────────────────────
# 0.  KONFIGURASI
# ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CLEAN_DIR   = BASE_DIR.parent / "data_clean"
RAW_PATH    = BASE_DIR / "datasoal.csv"
CLEAN_PATH  = CLEAN_DIR / "datasoal_clean.csv"
AUG_PATH    = CLEAN_DIR / "datasoal_augmented.csv"

TARGET_PER_TOPIK   = 60   # target minimum baris per topik
CRITICAL_THRESHOLD = 20   # topik < nilai ini → pakai AI augmentation
RULE_THRESHOLD     = 60   # topik < nilai ini → pakai rule-based

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

random.seed(42)

# ────────────────────────────────────────────────────────────
# 1.  PETA PENGGABUNGAN TOPIK
#     Format: "topik_lama" -> "topik_induk"
# ────────────────────────────────────────────────────────────
TOPIC_MERGE_MAP: dict[str, str] = {
    # Cahaya
    "sifat-sifat cahaya"                                        : "cahaya dan sifat-sifatnya",
    "merancang karya atau model dengan menerapkan sifat cahaya" : "cahaya dan sifat-sifatnya",

    # Tubuh manusia & hewan
    "organ tubuh manusia dan hewan" : "alat tubuh manusia dan hewan",
    "sistem pencernaan"             : "alat tubuh manusia dan hewan",
    "sistem pernapasan"             : "alat tubuh manusia dan hewan",
    "darah"                         : "alat tubuh manusia dan hewan",

    # Adaptasi makhluk hidup
    "penyesuaian diri makhluk hidup dengan lingkungannya" : "adaptasi makhluk hidup",

    # Benda dan sifatnya
    "bahan penyusun benda dan sifatnya" : "benda dan sifatnya",
    "perubahan sifat benda"             : "perubahan benda",

    # Gaya, gerak, energi
    "pengaruh gaya terhadap bentuk dan gerak suatu benda" : "gaya, gerak, dan energi",

    # Tumbuhan
    "pembuatan makanan pada tumbuhan hijau" : "tumbuhan hijau",

    # Bumi
    "struktur bumi"                        : "bumi dan peristiwa alam",
    "batuan dan proses pembentukan tanah"  : "bumi dan peristiwa alam",

    # SDA
    "air" : "sumber daya alam dan kegiatan manusia",
}


def apply_topic_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Ganti nama topik lama → topik induk sesuai TOPIC_MERGE_MAP."""
    df = df.copy()
    n_merged = 0
    for lama, induk in TOPIC_MERGE_MAP.items():
        mask = df["topik"] == lama
        if mask.any():
            df.loc[mask, "topik"] = induk
            n_merged += mask.sum()
    print(f"  ✓ {n_merged} baris dipindah ke topik induk berdasarkan TOPIC_MERGE_MAP")
    return df

# ────────────────────────────────────────────────────────────
# 2.  LOAD & CLEAN DATA
# ────────────────────────────────────────────────────────────
def load_clean() -> pd.DataFrame:
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
    else:
        df = pd.read_csv(RAW_PATH)
        df["topik"]    = df["topik"].str.lower().str.strip()
        df["subtopik"] = df["subtopik"].str.lower().str.strip()
        df = df.rename(columns={"link sumber buku": "link_sumber"}, errors="ignore")
        df = df.drop_duplicates(subset=["soal","jawaban"], keep="first").reset_index(drop=True)
        df["kelas"] = 5
        CLEAN_DIR.mkdir(exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False)

    if "jawaban_len"   not in df.columns: df["jawaban_len"]   = df["jawaban"].str.len()
    if "soal_len"      not in df.columns: df["soal_len"]      = df["soal"].str.len()
    if "is_augmented"  not in df.columns: df["is_augmented"]  = False
    if "aug_method"    not in df.columns: df["aug_method"]    = "original"

    # Terapkan penggabungan topik
    print("\n[TOPIC MERGE]")
    df = apply_topic_merge(df)
    return df

# ────────────────────────────────────────────────────────────
# 3.  ANALISIS
# ────────────────────────────────────────────────────────────
def analyze_sparse_topics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("topik").agg(
        jumlah   = ("soal", "count"),
        subtopik = ("subtopik", "nunique"),
        jw_len   = ("jawaban_len", "mean"),
    )
    stats["butuh_tambah"] = (TARGET_PER_TOPIK - stats["jumlah"]).clip(lower=0).astype(int)
    stats["metode"] = stats["jumlah"].apply(
        lambda n: "AI" if n < CRITICAL_THRESHOLD else
                  "rule-based" if n < RULE_THRESHOLD else "tidak perlu"
    )
    return stats[stats["butuh_tambah"] > 0].sort_values("jumlah")

# ────────────────────────────────────────────────────────────
# 4.  RULE-BASED AUGMENTATION — TEKNIK DIPERLUAS
# ────────────────────────────────────────────────────────────

# ── 4a. Kosakata ──────────────────────────────────────────
VARIASI_KATA_TANYA = {
    "apa"           : ["sebutkan", "tuliskan", "jelaskan apa yang dimaksud dengan",
                       "deskripsikan", "definisikan"],
    "sebutkan"      : ["apa saja", "tuliskan", "berikan contoh", "rincikan", "daftarkan"],
    "jelaskan"      : ["uraikan", "ceritakan", "bagaimana", "deskripsikan",
                       "paparkan", "terangkan"],
    "bagaimana"     : ["jelaskan bagaimana", "ceritakan", "uraikan", "terangkan proses"],
    "mengapa"       : ["kenapa", "apa alasan", "apa penyebab", "apa yang membuat",
                       "faktor apa yang menyebabkan"],
    "berikan contoh": ["sebutkan contoh", "tuliskan contoh", "apa saja contoh",
                       "berikan 2 contoh"],
    "tuliskan"      : ["sebutkan", "apa saja", "berikan", "daftarkan"],
    "apakah"        : ["benarkah bahwa", "benar atau salah", "apa itu"],
}

SINONIM = {
    "manusia"     : ["tubuh kita", "orang", "makhluk manusia"],
    "hewan"       : ["binatang", "fauna", "makhluk hidup"],
    "tumbuhan"    : ["tanaman", "flora", "tumbuh-tumbuhan"],
    "berguna"     : ["bermanfaat", "berfungsi", "berperan"],
    "perubahan"   : ["pergeseran", "transformasi", "pergantian"],
    "proses"      : ["tahapan", "cara kerja", "mekanisme"],
    "membutuhkan" : ["memerlukan", "butuh", "bergantung pada"],
    "menghasilkan": ["memproduksi", "membuat", "menciptakan"],
    "melindungi"  : ["menjaga", "mempertahankan", "membentengi"],
    "berfungsi"   : ["berperan", "bekerja sebagai", "berguna"],
    "mempengaruhi": ["berdampak pada", "berpengaruh terhadap", "mengubah"],
    "terdiri dari": ["tersusun atas", "terbuat dari", "dibentuk oleh"],
    "contoh"      : ["misalnya", "antara lain", "seperti"],
}

KONTEKS_PREFIX = [
    "dalam kehidupan sehari-hari, ",
    "di sekolah kita mempelajari bahwa ",
    "penting untuk diketahui bahwa ",
    "berdasarkan materi IPA kelas 5, ",
    "sebagai siswa kelas 5, kita perlu tahu bahwa ",
    "dalam ilmu pengetahuan alam, ",
    "perlu dipahami bahwa ",
    "secara ilmiah, ",
]

INTRO_SOAL = [
    "coba ",
    "dapatkah kamu ",
    "tahukah kamu ",
    "apakah kamu bisa ",
    "menurutmu, ",
    "berdasarkan yang kamu pelajari, ",
    "dengan kata-katamu sendiri, ",
]

# ── 4b. Transformasi soal ──────────────────────────────────

def variasi_kata_tanya(soal: str) -> list[str]:
    """Ganti kata tanya pembuka dengan sinonimnya."""
    hasil = []
    soal_lower = soal.lower().strip().rstrip("!?.")
    for kata, variasi_list in VARIASI_KATA_TANYA.items():
        if soal_lower.startswith(kata):
            sisa = soal_lower[len(kata):].strip()
            for var in variasi_list:
                ns = f"{var} {sisa}?"
                ns = ns[0].upper() + ns[1:]
                if ns.lower() != soal.lower():
                    hasil.append(ns)
            break
    return hasil


def ganti_sinonim(soal: str) -> list[str]:
    """Ganti satu kata dengan sinonimnya → menghasilkan beberapa variasi."""
    hasil = []
    for kata, syn_list in SINONIM.items():
        if kata in soal.lower():
            for syn in syn_list[:2]:
                ns = re.sub(re.escape(kata), syn, soal, count=1, flags=re.IGNORECASE)
                if ns != soal:
                    hasil.append(ns[0].upper() + ns[1:])
    return hasil


def tambah_intro(soal: str) -> list[str]:
    """Tambah frasa pembuka di depan soal."""
    hasil = []
    soal_lower = soal.lower().rstrip("!?.")
    for intro in INTRO_SOAL[:3]:
        alt = intro + soal_lower + "?"
        alt = alt[0].upper() + alt[1:]
        if alt.lower() != soal.lower():
            hasil.append(alt)
    return hasil


def ubah_ke_pertanyaan_pilihan(soal: str, jawaban: str) -> str | None:
    """Ubah soal terbuka menjadi konfirmasi ya/tidak."""
    soal_lower = soal.lower().strip()
    for kt in ["apa", "sebutkan", "tuliskan", "jelaskan", "bagaimana", "mengapa"]:
        if soal_lower.startswith(kt):
            sisa = soal_lower[len(kt):].strip().rstrip("!?.")
            ns = f"Benarkah bahwa {sisa}? Jelaskan!"
            return ns[0].upper() + ns[1:]
    return None


def ubah_ke_pasif(soal: str) -> str | None:
    """Ubah kalimat aktif ke bentuk pasif sederhana (heuristik)."""
    pola = [
        (r"bagaimana (\w+ )?(\w+) (me\w+) (\w+)", r"bagaimana \4 di\3 oleh \1\2"),
        (r"apa yang (me\w+) (\w+)", r"apa yang menyebabkan \2 terjadi"),
    ]
    for pat, rep in pola:
        ns = re.sub(pat, rep, soal, flags=re.IGNORECASE)
        if ns != soal:
            return ns[0].upper() + ns[1:]
    return None


def variasi_konteks(konteks: str) -> str:
    """Ganti prefiks konteks secara acak."""
    prefix_cycle = cycle(KONTEKS_PREFIX)
    for p in prefix_cycle:
        stripped = konteks.strip()
        for ep in KONTEKS_PREFIX:
            if stripped.lower().startswith(ep):
                stripped = stripped[len(ep):]
                break
        if stripped:
            return p + stripped[0].lower() + stripped[1:]
    return konteks


def semua_variasi_soal(soal: str, jawaban: str) -> list[tuple[str, str]]:
    """
    Kumpulkan semua variasi soal dari semua teknik.
    Return: list of (soal_baru, metode_label)
    """
    kandidat = []
    for s in variasi_kata_tanya(soal):   kandidat.append((s, "kata_tanya"))
    for s in ganti_sinonim(soal):        kandidat.append((s, "sinonim"))
    for s in tambah_intro(soal):         kandidat.append((s, "intro"))
    s = ubah_ke_pertanyaan_pilihan(soal, jawaban)
    if s: kandidat.append((s, "konfirmasi"))
    s = ubah_ke_pasif(soal)
    if s: kandidat.append((s, "pasif"))
    # deduplikasi
    seen = set()
    result = []
    for s, m in kandidat:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append((s, m))
    return result


# ── 4c. Fungsi utama rule-based ────────────────────────────

def rule_based_augment(df: pd.DataFrame, topik: str, n_target: int) -> list[dict]:
    existing   = df[df["topik"] == topik].copy()
    n_current  = len(existing)
    n_needed   = max(0, n_target - n_current)
    if n_needed == 0:
        return []

    new_rows      = []
    existing_soal = set(df["soal"].str.lower().str.strip())

    # Acak urutan agar tiap run menghasilkan kombinasi berbeda
    rows = existing.sample(frac=1, random_state=42).to_dict("records")
    row_cycle = cycle(rows)

    teknik_counter: dict[str, int] = {}

    for _ in range(n_needed * 10):          # batas iterasi aman
        if len(new_rows) >= n_needed:
            break
        row = next(row_cycle)
        variasi = semua_variasi_soal(row["soal"], row["jawaban"])
        random.shuffle(variasi)

        for vs, metode in variasi:
            if len(new_rows) >= n_needed:
                break
            if vs.lower().strip() in existing_soal:
                continue
            nr = row.copy()
            nr["soal"]         = vs
            nr["konteks"]      = variasi_konteks(row.get("konteks", ""))
            nr["soal_len"]     = len(vs)
            nr["is_augmented"] = True
            nr["aug_method"]   = f"rule:{metode}"
            new_rows.append(nr)
            existing_soal.add(vs.lower().strip())
            teknik_counter[metode] = teknik_counter.get(metode, 0) + 1

    print(f"  ✓ Rule-based: {len(new_rows)} baris baru")
    print(f"     Distribusi teknik: " +
          ", ".join(f"{k}={v}" for k, v in sorted(teknik_counter.items())))
    return new_rows

# ────────────────────────────────────────────────────────────
# 5.  AI-BASED AUGMENTATION (Claude API)
# ────────────────────────────────────────────────────────────

def call_claude_api(prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY tidak ditemukan.")

    payload = json.dumps({
        "model"     : "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages"  : [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {
            "Content-Type"      : "application/json",
            "x-api-key"         : ANTHROPIC_API_KEY,
            "anthropic-version" : "2023-06-01",
        },
        method = "POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def build_ai_prompt(topik: str, subtopik_list: list, examples: list, n_generate: int) -> str:
    examples_text = "\n".join([
        f"  subtopik : {e['subtopik']}\n"
        f"  soal     : {e['soal']}\n"
        f"  jawaban  : {e['jawaban']}\n"
        f"  contoh   : {e.get('contoh','')}\n"
        f"  konteks  : {e.get('konteks','')}\n"
        for e in examples[:4]
    ])
    subtopik_text = ", ".join(subtopik_list)

    return f"""Kamu adalah pakar pendidikan IPA Sekolah Dasar Kelas 5 Indonesia.
Tugas: generate tepat {n_generate} pasangan soal-jawaban BARU untuk topik berikut.

TOPIK    : {topik}
SUBTOPIK : {subtopik_text}

CONTOH DATA (referensi gaya & tingkat kesulitan):
{examples_text}

ATURAN WAJIB:
1. Sesuai kurikulum IPA SD Kelas 5 (BSE Kemdikbud).
2. Jawaban akurat secara ilmiah, bahasa mudah dipahami siswa SD.
3. "konteks" = kalimat latar belakang singkat (1 kalimat).
4. "contoh" = contoh nyata atau analogi sederhana.
5. Variasikan SEMUA jenis berikut secara merata:
   - Jenis kata tanya: apa, bagaimana, mengapa, sebutkan, jelaskan, bandingkan
   - Tingkat kognitif: ingatan (C1), pemahaman (C2), penerapan (C3)
   - Format soal: deskriptif, konfirmasi (benarkah...), sebab-akibat, perbandingan
6. JANGAN duplikasi soal contoh di atas.
7. Pastikan mencakup minimal {min(3, len(subtopik_list))} subtopik berbeda.

KEMBALIKAN HANYA JSON valid (tidak ada teks lain):
[
  {{
    "subtopik": "...",
    "soal": "...",
    "jawaban": "...",
    "contoh": "...",
    "konteks": "..."
  }},
  ...
]"""


def ai_based_augment(df: pd.DataFrame, topik: str, n_target: int) -> list[dict]:
    existing  = df[df["topik"] == topik]
    n_current = len(existing)
    n_needed  = max(0, n_target - n_current)
    if n_needed == 0 or not ANTHROPIC_API_KEY:
        return []

    subtopik_list = existing["subtopik"].unique().tolist()
    examples      = existing[["subtopik","soal","jawaban","contoh","konteks"]]\
                    .sample(min(4, len(existing)), random_state=42).to_dict("records")

    print(f"   → Memanggil Claude API untuk {n_needed} data baru ...")
    try:
        prompt    = build_ai_prompt(topik, subtopik_list, examples, n_needed)
        response  = call_claude_api(prompt)
        clean     = re.sub(r"```json|```", "", response).strip()
        generated = json.loads(clean)

        link_ref  = existing["link_sumber"].iloc[0] if "link_sumber" in existing else ""
        new_rows  = []
        for item in generated:
            row = {
                "no"          : 0,
                "topik"       : topik,
                "subtopik"    : item.get("subtopik", subtopik_list[0]),
                "soal"        : item.get("soal", ""),
                "jawaban"     : item.get("jawaban", ""),
                "contoh"      : item.get("contoh", ""),
                "konteks"     : item.get("konteks", ""),
                "link_sumber" : link_ref,
                "kelas"       : 5,
                "soal_len"    : len(item.get("soal","")),
                "jawaban_len" : len(item.get("jawaban","")),
                "is_augmented": True,
                "aug_method"  : "ai:claude",
            }
            new_rows.append(row)

        print(f"   ✓ Berhasil generate {len(new_rows)} pasangan soal-jawaban baru")
        return new_rows

    except json.JSONDecodeError as e:
        print(f"   ✗ Gagal parse JSON: {e} → fallback rule-based")
        return rule_based_augment(df, topik, n_target)
    except Exception as e:
        print(f"   ✗ Error API: {e} → fallback rule-based")
        return rule_based_augment(df, topik, n_target)

# ────────────────────────────────────────────────────────────
# 6.  MAIN
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("AUGMENTASI DATASET — CC26-PSU312  (v2 — Topic Merge + Varied)")
    print("=" * 65)

    df = load_clean()
    print(f"\n✓ Data dimuat: {len(df)} baris, {df['topik'].nunique()} topik unik")
    print(f"  (setelah penggabungan topik serupa)\n")

    # Laporan penggabungan topik
    if TOPIC_MERGE_MAP:
        print("  Topik yang DIGABUNGKAN:")
        merged_topics: dict[str, list] = {}
        for lama, induk in TOPIC_MERGE_MAP.items():
            merged_topics.setdefault(induk, []).append(lama)
        for induk, lamas in sorted(merged_topics.items()):
            print(f"    [{induk}]  ← {', '.join(lamas)}")
        print()

    sparse = analyze_sparse_topics(df)

    print(f"{'─'*65}")
    print(f"  {'Topik':<48} {'Soal':>4} {'Perlu':>5}  {'Metode'}")
    print(f"{'─'*65}")
    for topik, row in sparse.iterrows():
        print(f"  {topik:<48} {int(row['jumlah']):>4} {int(row['butuh_tambah']):>5}  {row['metode']}")
    print(f"{'─'*65}")
    print(f"  Total data baru yang perlu di-generate: {sparse['butuh_tambah'].sum()}\n")

    if not ANTHROPIC_API_KEY:
        print("⚠  ANTHROPIC_API_KEY tidak ditemukan — topik kritis akan pakai rule-based.\n")

    # ── Augmentasi ──────────────────────────────────────────
    all_new_rows = []
    for topik, row in sparse.iterrows():
        n_target = int(row["jumlah"]) + int(row["butuh_tambah"])
        print(f"\n[{row['metode'].upper()}] {topik}")
        print(f"  {int(row['jumlah'])} baris → target {n_target} baris")

        if row["metode"] == "AI" and ANTHROPIC_API_KEY:
            new_rows = ai_based_augment(df, topik, n_target)
            time.sleep(1)
        else:
            new_rows = rule_based_augment(df, topik, n_target)

        all_new_rows.extend(new_rows)

    # ── Gabungkan & simpan ───────────────────────────────────
    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows)
        for col in df.columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_aug = pd.concat([df, df_new[df.columns]], ignore_index=True)
    else:
        df_aug = df.copy()
        print("\n⚠  Tidak ada baris baru yang berhasil di-generate.")

    df_aug["no"] = range(1, len(df_aug) + 1)
    df_aug = df_aug.drop_duplicates(subset=["soal","jawaban"], keep="first").reset_index(drop=True)

    CLEAN_DIR.mkdir(exist_ok=True)
    df_aug.to_csv(AUG_PATH, index=False, encoding="utf-8")

    # ── Laporan akhir ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("LAPORAN AUGMENTASI")
    print("=" * 65)
    print(f"  Data sebelum : {len(df):>5} baris ({df['topik'].nunique()} topik)")
    n_ai   = (df_aug["aug_method"] == "ai:claude").sum() if "aug_method" in df_aug else 0
    n_rule = df_aug["aug_method"].str.startswith("rule:").sum() if "aug_method" in df_aug else 0
    print(f"  Data baru    : {len(df_aug)-len(df):>5} baris  "
          f"(AI={n_ai}, rule-based={n_rule})")
    print(f"  Data sesudah : {len(df_aug):>5} baris ({df_aug['topik'].nunique()} topik)")
    print(f"\n  Disimpan ke  : {AUG_PATH}")

    # Distribusi teknik rule-based
    if "aug_method" in df_aug.columns:
        rule_methods = df_aug[df_aug["aug_method"].str.startswith("rule:", na=False)]["aug_method"].value_counts()
        if not rule_methods.empty:
            print("\n  Distribusi teknik rule-based:")
            for m, cnt in rule_methods.items():
                print(f"    {m:<25} {cnt:>4}")

    print("\nStatus per topik sesudah augmentasi:")
    stats_after = df_aug.groupby("topik")["soal"].count().sort_values(ascending=False)
    for t, cnt in stats_after.items():
        flag = "✓" if cnt >= TARGET_PER_TOPIK else "⚠" if cnt >= 20 else "✗"
        bar  = "█" * min(int(cnt / 3), 25)
        print(f"  {flag} {t:<52} {cnt:>4}  {bar}")

    print("\n" + "=" * 65)
    print("✅  Augmentasi selesai!")
    print(f"   Lanjut: ganti path di dashboard.py → datasoal_augmented.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()