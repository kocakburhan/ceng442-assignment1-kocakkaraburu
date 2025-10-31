# -*- coding: utf-8 -*-
# process_data.py
# Implemented domain-aware text preprocessing with normalization and sentiment mapping for Azerbaijani

import re
import html
import unicodedata
import pandas as pd
from pathlib import Path

# Used ftfy for text fixing if available
try:
    from ftfy import fix_text
except Exception:
    def fix_text(s): return s

# --- Domain Detection and Normalization ---

# Domain detection patterns
NEWS_HINTS = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dhaaa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#", re.I)
REV_HINTS = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

# Normalization patterns for reviews domain
PRICE_RE = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE = re.compile(r"\bçox yaxşı\b")
NEG_RATE = re.compile(r"\bçox pis\b")

def detect_domain(text: str) -> str:
    """Detected text domain using pattern matching."""
    if not isinstance(text, str):
        return "general"
    s = text.lower()
    if NEWS_HINTS.search(s): return "news"
    if SOCIAL_HINTS.search(s): return "social"
    if REV_HINTS.search(s): return "reviews"
    return "general"

def domain_specific_normalize(cleaned: str, domain: str) -> str:
    """Applied domain-specific normalization (e.g., tokenizing prices and ratings for reviews)."""
    s = cleaned
    if domain == "reviews":
        s = PRICE_RE.sub(" <PRICE> ", s)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        return " ".join(s.split())
    return cleaned

def add_domain_tag(line: str, domain: str) -> str:
    """Added domain tag prefix to corpus lines."""
    return f"dom{domain} " + line

# --- Text Normalization Pipeline ---

def lower_az(s: str) -> str:
    """Implemented Azerbaijani-specific lowercasing."""
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("I", "ı").replace("İ", "i")
    s = s.lower().replace("i̇", "i")
    return s

# Text cleaning patterns
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?[\d\s\-\(\)]{6,}\d")
USER_RE = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

# Mapped emojis to sentiment tokens
EMO_MAP = {
    "😊": " EMO_POS ", "🙂": " EMO_POS ", "👍": " EMO_POS ", "❤": " EMO_POS ", "😍": " EMO_POS ",
    "😒": " EMO_NEG ", "☹": " EMO_NEG ", "😡": " EMO_NEG ", "👎": " EMO_NEG ", "😞": " EMO_NEG "
}
# Normalized common slang terms
SLANG_MAP = {"sim":"salam", "tmm":"tamam", "sagol":"sağol", "cox":"çox", "yaxsi":"yaxşı"}
# Negation words for scope marking
NEGATORS = {"yox", "deyil", "heç", "qətiyyən", "yoxdur"}

def normalize_text_az(s: str, numbers_to_token=True, keep_sentence_punct=False) -> str:
    """Implemented comprehensive text normalization for Azerbaijani."""
    if not isinstance(s, str): return ""

    # Replaced emojis with sentiment tokens
    for emo, tag in EMO_MAP.items():
        s = s.replace(emo, f" {tag} ")
        
    s = fix_text(s)
    s = html.unescape(s)
    s = HTML_TAG_RE.sub("", s)
    
    # Tokenized URLs, emails, phones, and user mentions
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    s = USER_RE.sub(" USER ", s)
    
    # Split hashtags by camelCase
    s = re.sub(r"#([A-Za-z0-9_]+)", lambda m: " " + re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1)) + " ", s)
    
    s = lower_az(s)
    s = MULTI_PUNCT.sub(r"\1", s)
    
    if numbers_to_token:
        s = re.sub(r"\d+", " <NUM> ", s)
        
    if keep_sentence_punct:
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ.!?]", "", s)
    else:
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", "", s)
        
    s = MULTI_SPACE.sub(" ", s).strip()
    
    # Token-level processing with negation scope
    toks = s.split()
    norm = []
    mark_neg = 0
    
    for t in toks:
        t = REPEAT_CHARS.sub(r"\1\1", t)
        t = SLANG_MAP.get(t, t)
        
        # Applied negation scope marking
        if t in NEGATORS:
            norm.append(t)
            mark_neg = 3
            continue
            
        if mark_neg > 0 and t not in {"URL", "EMAIL", "PHONE", "USER", "<NUM>"}:
            norm.append(t + "_NEG")
            mark_neg -= 1
        else:
            norm.append(t)
            mark_neg = 0
            
    # Removed single-character tokens except 'o' and 'e'
    norm = [t for t in norm if not (len(t) == 1 and t not in ("o", "e"))]
    return " ".join(norm).strip()

def map_sentiment_value(v, scheme: str):
    """Mapped various sentiment labels to 0.0, 0.5, 1.0 values."""
    if scheme == "binary":
        try: return 1.0 if int(v) == 1 else 0.0
        except Exception: pass
    
    s = str(v).strip().lower()
    if s in ("pos", "positive", "1", "müsbət", "good", "pozitiv"): return 1.0
    if s in ("neu", "neutral", "2", "neytral"): return 0.5
    if s in ("neg", "negative", "0", "mənfi", "bad", "negativ"): return 0.0
    return None

def process_file(in_path, text_col, label_col, scheme, out_two_col_path, remove_stopwords=False):
    """Processed Excel file with text cleaning and sentiment mapping."""
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
    df_original = df.copy()
    
    # Applied base text cleaning
    df["cleaned_text"] = df[text_col].astype(str).apply(lambda s: normalize_text_az(s))
    
    # Detected domain and applied domain-specific normalization
    df["domain"] = df_original[text_col].astype(str).apply(detect_domain)
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["domain"]), axis=1)
    
    # Optional stopword removal (disabled by default)
    if remove_stopwords:
        SW = set(["və", "ilə","amma","ancaq", "lakin", "ya", "həm", "ki", "bu", "bir", "o", "biz", "siz", "men"
 ,"sən", "orada", "burada", "bütün", "hər", "artıq", "çox", "az", "ən", "də", "da", "üçün"])
        for keep in NEGATORS:
            SW.discard(keep)
        df["cleaned_text"] = df["cleaned_text"].apply(lambda s: " ".join([t for t in s.split() if t not in SW]))

    # Mapped sentiment labels to numeric values
    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    df["sentiment_value"] = df["sentiment_value"].astype(float)
    
    # Removed duplicates and empty entries
    df = df.drop_duplicates(subset=["cleaned_text"])
    df = df[df["cleaned_text"].str.len() > 0]
    
    # Saved final 2-column output
    out_df = df[["cleaned_text", "sentiment_value"]].reset_index(drop=True)
    Path(out_two_col_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_two_col_path, index=False)
    print(f"  Saved: {out_two_col_path} (rows={len(out_df)})")

def build_corpus_txt(input_files, text_cols, out_txt="corpus_all.txt"):
    """Built domain-tagged corpus file from all datasets."""
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
            # Split into sentences
            s = normalize_text_az(raw, keep_sentence_punct=True)
            parts = re.split(r"[.!?]+", s)
            for p in parts:
                if not p.strip(): continue
                # Removed punctuation from sentences
                p = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", "", p)
                p = MULTI_SPACE.sub(" ", p).strip().lower()
                if p:
                    lines.append(add_domain_tag(p, dom))

    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")
    print(f"Wrote {out_txt} with {len(lines)} lines")

# --- Main Execution ---
if __name__ == "__main__":
    
    # Configuration for 5 datasets
    CFG = [
        ("labeled-sentiment.xlsx", "text", "sentiment", "tri"),
        ("test__1_.xlsx", "text", "label", "binary"),
        ("train__3_.xlsx", "text", "label", "binary"),
        ("train-00000-of-00001.xlsx", "text", "labels", "tri"),
        ("merged_dataset_CSV__1_.xlsx", "text", "labels", "binary"),
    ]
    
    print("--- Step 1: Processing files into 2-column format ---")
    # Generated 2-column Excel outputs
    input_files_for_corpus = []
    input_cols_for_corpus = []
    
    for fname, tcol, lcol, scheme in CFG:
        out_f = f"output/{(Path(fname).stem)}_2col.xlsx"
        process_file(fname, tcol, lcol, scheme, out_f, remove_stopwords=False)
        input_files_for_corpus.append(fname)
        input_cols_for_corpus.append(tcol)

    print("\n--- Step 2: Building combined corpus_all.txt ---")
    # Created combined corpus file
    build_corpus_txt(input_files_for_corpus, input_cols_for_corpus, out_txt="output/corpus_all.txt")
    
    print("\nData processing complete.")