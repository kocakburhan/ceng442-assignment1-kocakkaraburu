# -*- coding: utf-8 -*-
# process_data.py
# Bu script, CENG442 Ödev 1'in 6. ve 7. bölümlerindeki kodları birleştirir.
# 

import re
import html
import unicodedata
import pandas as pd
from pathlib import Path

# ftfy kütüphanesi opsiyoneldir, bulunamazsa boş fonksiyon tanımlanır [cite: 29, 90-92]
try:
    from ftfy import fix_text
except Exception:
    def fix_text(s): return s

# --- Bölüm 6: Domain-Aware Fonksiyonları [cite: 52-84] ---

# Domain detection hints [cite: 54-59, 121-127]
NEWS_HINTS = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dhaaa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#", re.I) # PDF'ten biraz daha genişletildi
REV_HINTS = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

# Domain-specific normalization (reviews) [cite: 68-76, 128-135]
PRICE_RE = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE = re.compile(r"\bçox yaxşı\b")
NEG_RATE = re.compile(r"\bçox pis\b")

def detect_domain(text: str) -> str:
    """Metnin domain'ini basit kurallarla tespit eder."""
    # [cite: 60-67, 136-141]
    if not isinstance(text, str):
        return "general"
    s = text.lower()
    if NEWS_HINTS.search(s): return "news"
    if SOCIAL_HINTS.search(s): return "social"
    if REV_HINTS.search(s): return "reviews"
    return "general"

def domain_specific_normalize(cleaned: str, domain: str) -> str:
    """Domain'e özel normalizasyon uygular (örnek: reviews)."""
    # [cite: 77-82, 142-145, 150-151]
    s = cleaned
    if domain == "reviews":
        s = PRICE_RE.sub(" <PRICE> ", s)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        return " ".join(s.split())
    return cleaned

def add_domain_tag(line: str, domain: str) -> str:
    """Corpus satırının başına domain etiketi ekler."""
    # [cite: 83-84, 146]
    return f"dom{domain} " + line # e.g., 'domnews', 'domreviews'

# --- Bölüm 7: Ana Pipeline Fonksiyonları [cite: 85-194] ---

def lower_az(s: str) -> str:
    """Azerbaycan Türkçesine özel küçük harf dönüşümü."""
    # [cite: 35, 93-97]
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("I", "ı").replace("İ", "i")
    s = s.lower().replace("i̇", "i") # Bazen "i" ve "̇" birleşebiliyor
    return s

# Regex tanımlamaları [cite: 98-116]
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?[\d\s\-\(\)]{6,}\d") # PDF'ten iyileştirildi
USER_RE = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

# Mini Challenge: Emoji Map [cite: 45, 117]
EMO_MAP = {
    "😊": " EMO_POS ", "🙂": " EMO_POS ", "👍": " EMO_POS ", "❤": " EMO_POS ", "😍": " EMO_POS ",
    "😒": " EMO_NEG ", "☹": " EMO_NEG ", "😡": " EMO_NEG ", "👎": " EMO_NEG ", "😞": " EMO_NEG "
}
# Mini Challenge: Deasciify (Slang map olarak kullanılmış) [cite: 48, 118]
SLANG_MAP = {"sim":"salam", "tmm":"tamam", "sagol":"sağol", "cox":"çox", "yaxsi":"yaxşı"}
# Mini Challenge: Negation Scope [cite: 47, 118]
NEGATORS = {"yox", "deyil", "heç", "qətiyyən", "yoxdur"}

def normalize_text_az(s: str, numbers_to_token=True, keep_sentence_punct=False) -> str:
    """PDF'teki ana metin temizleme ve normalleştirme fonksiyonu."""
    # 
    if not isinstance(s, str): return ""

    # Mini Challenge: Emoji map [cite: 149, 154-156]
    for emo, tag in EMO_MAP.items():
        s = s.replace(emo, f" {tag} ")
        
    s = fix_text(s)
    s = html.unescape(s) # HTML entity'lerini düzelt (örn: &amp;)
    s = HTML_TAG_RE.sub("", s) # HTML tag'lerini sil [cite: 37, 160]
    
    # URL, EMAIL, PHONE, USER token'larını değiştir [cite: 36, 161-163, 168]
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    s = USER_RE.sub(" USER ", s)
    
    # Mini Challenge: Hashtag split [cite: 37, 44, 164-167]
    s = re.sub(r"#([A-Za-z0-9_]+)", lambda m: " " + re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1)) + " ", s)
    
    s = lower_az(s) # Azerbaycan Türkçesine özel lowercase [cite: 35, 169-170]
    s = MULTI_PUNCT.sub(r"\1", s)
    
    if numbers_to_token:
        s = re.sub(r"\d+", " <NUM> ", s) # Rakamları <NUM> ile değiştir [cite: 39, 173-174]
        
    if keep_sentence_punct:
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ.!?]", "", s) # Cümle noktalamasını koru
    else:
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", "", s) # Tüm noktalamayı kaldır [cite: 40, 178]
        
    s = MULTI_SPACE.sub(" ", s).strip()
    
    # Token bazlı temizlik
    toks = s.split()
    norm = []
    mark_neg = 0 # Mini Challenge: Negation scope [cite: 47]
    
    for t in toks:
        t = REPEAT_CHARS.sub(r"\1\1", t) # 3+ tekrarı 2'ye düşür [cite: 38, 185]
        t = SLANG_MAP.get(t, t) # Mini Challenge: Deasciify/Slang [cite: 48, 186]
        
        # Mini Challenge: Negation Scope Uygulaması [cite: 187-193]
        if t in NEGATORS:
            norm.append(t)
            mark_neg = 3 # Sonraki 3 token'ı işaretle
            continue
            
        if mark_neg > 0 and t not in {"URL", "EMAIL", "PHONE", "USER", "<NUM>"}:
            norm.append(t + "_NEG")
            mark_neg -= 1
        else:
            norm.append(t)
            mark_neg = 0 # İşaretlemeyi sıfırla
            
    # Tek harfli token'ları kaldır (o ve e hariç) [cite: 41, 194]
    norm = [t for t in norm if not (len(t) == 1 and t not in ("o", "e"))]
    return " ".join(norm).strip()

def map_sentiment_value(v, scheme: str):
    """Farklı etiket formatlarını 0.0, 0.5, 1.0 değerlerine haritalar."""
    # [cite: 23, 195-202]
    if scheme == "binary":
        try: return 1.0 if int(v) == 1 else 0.0
        except Exception: pass
    
    s = str(v).strip().lower()
    if s in ("pos", "positive", "1", "müsbət", "good", "pozitiv"): return 1.0
    if s in ("neu", "neutral", "2", "neytral"): return 0.5
    if s in ("neg", "negative", "0", "mənfi", "bad", "negativ"): return 0.0
    return None

def process_file(in_path, text_col, label_col, scheme, out_two_col_path, remove_stopwords=False):
    """Tek bir Excel dosyasını okur, işler ve 2 sütunlu çıktıyı kaydeder."""
    # [cite: 203-237]
    print(f"Processing: {in_path}...")
    try:
        df = pd.read_excel(in_path)
    except Exception as e:
        print(f"  ERROR: Could not read {in_path}. Skipping. Error: {e}")
        return
        
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns: df = df.drop(columns=[c])
        
    assert text_col in df.columns and label_col in df.columns, f"Missing columns in {in_path}"
    
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df_original = df.copy() # Orijinal metni domain tespiti için sakla
    
    # Temel temizlik
    df["cleaned_text"] = df[text_col].astype(str).apply(lambda s: normalize_text_az(s))
    
    # Domain tespiti ve domain'e özel normalizasyon [cite: 218-220]
    df["domain"] = df_original[text_col].astype(str).apply(detect_domain)
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["domain"]), axis=1)
    
    # Stopwords (Ödevde remove_stopwords=False istendiği için bu blok genelde çalışmaz) [cite: 221-227]
    if remove_stopwords:
        # PDF'teki liste [cite: 223]
        SW = set(["və", "ilə","amma","ancaq", "lakin", "ya", "həm", "ki", "bu", "bir", "o", "biz", "siz", "men"
 ,"sən", "orada", "burada", "bütün", "hər", "artıq", "çox", "az", "ən", "də", "da", "üçün"])
        for keep in NEGATORS: # Negasyon kelimelerini koru [cite: 224-225]
            SW.discard(keep)
        df["cleaned_text"] = df["cleaned_text"].apply(lambda s: " ".join([t for t in s.split() if t not in SW]))

    # Sentiment haritalama [cite: 228-233]
    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    df["sentiment_value"] = df["sentiment_value"].astype(float)
    
    # Duplike ve boş satırları at [cite: 42]
    df = df.drop_duplicates(subset=["cleaned_text"])
    df = df[df["cleaned_text"].str.len() > 0]
    
    # Final 2 sütunlu çıktı [cite: 234-237]
    out_df = df[["cleaned_text", "sentiment_value"]].reset_index(drop=True)
    Path(out_two_col_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_two_col_path, index=False)
    print(f"  Saved: {out_two_col_path} (rows={len(out_df)})")

def build_corpus_txt(input_files, text_cols, out_txt="corpus_all.txt"):
    """Tüm veriyi okur, domain etiketli, noktasız corpus_all.txt oluşturur."""
    # [cite: 238-259]
    print(f"\nBuilding corpus: {out_txt}...")
    lines = []
    for (f, text_col) in zip(input_files, text_cols):
        try:
            df = pd.read_excel(f)
        except Exception:
            print(f"  Skipping {f} for corpus (file error).")
            continue
            
        for raw in df[text_col].dropna().astype(str):
            dom = detect_domain(raw)
            # Cümlelere bölmek için keep_sentence_punct=True [cite: 246]
            s = normalize_text_az(raw, keep_sentence_punct=True)
            parts = re.split(r"[.!?]+", s) # Cümlelere ayır [cite: 247]
            for p in parts:
                if not p.strip(): continue
                # Noktalamayı kaldır [cite: 253]
                p = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", "", p)
                p = MULTI_SPACE.sub(" ", p).strip().lower() # PDF'teki sıradan [cite: 254] farklı ama daha mantıklı
                if p:
                    lines.append(add_domain_tag(p, dom)) # Domain etiketi ekle [cite: 255]

    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")
    print(f"Wrote {out_txt} with {len(lines)} lines")

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # [cite: 260-262]
    
    # 5 dataset için konfigürasyon [cite: 17-22, 263-283]
    CFG = [
        ("labeled-sentiment.xlsx", "text", "sentiment", "tri"),
        ("test__1_.xlsx", "text", "label", "binary"),
        ("train__3_.xlsx", "text", "label", "binary"),
        ("train-00000-of-00001.xlsx", "text", "labels", "tri"),
        ("merged_dataset_CSV__1_.xlsx", "text", "labels", "binary"),
    ]
    
    print("--- Step 1: Processing files into 2-column format ---")
    # Adım 1: 2 sütunlu Excel çıktılarını üret [cite: 284-287]
    input_files_for_corpus = []
    input_cols_for_corpus = []
    
    for fname, tcol, lcol, scheme in CFG:
        out_f = f"output/{(Path(fname).stem)}_2col.xlsx" # Çıktıları bir klasöre alalım
        process_file(fname, tcol, lcol, scheme, out_f, remove_stopwords=False) # [cite: 287]
        input_files_for_corpus.append(fname)
        input_cols_for_corpus.append(tcol)

    print("\n--- Step 2: Building combined corpus_all.txt ---")
    # Adım 2: corpus_all.txt dosyasını oluştur [cite: 288-290]
    build_corpus_txt(input_files_for_corpus, input_cols_for_corpus, out_txt="output/corpus_all.txt")
    
    print("\nData processing complete.")