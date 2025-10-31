# evaluate_models.py
# Evaluated Word2Vec and FastText models using lexical coverage, similarity metrics, and nearest neighbors

import pandas as pd
from gensim.models import Word2Vec, FastText
from numpy import dot
from numpy.linalg import norm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

print("--- Step 4: Evaluating embedding models ---")

# Loaded trained models
try:
    w2v = Word2Vec.load("embeddings/word2vec.model")
    ft = FastText.load("embeddings/fasttext.model")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Could not load models. Did you run train_models.py? Error: {e}")
    exit()

# --- 1. Lexical Coverage ---
def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

files = [
    "output/labeled-sentiment_2col.xlsx",
    "output/test__1__2col.xlsx",
    "output/train__3__2col.xlsx",
    "output/train-00000-of-00001_2col.xlsx",
    "output/merged_dataset_CSV__1__2col.xlsx",
]

def read_tokens(f):
    df = pd.read_excel(f, usecols=["cleaned_text"])
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

print("\n== 1. Lexical Coverage (per dataset) ==")
all_toks = []
for f in files:
    f_path = Path(f)
    if not f_path.exists():
        print(f"  Skipping coverage for {f_path.name} (file not found).")
        continue
    toks = read_tokens(f)
    all_toks.extend(toks)
    cov_w2v = lexical_coverage(w2v, toks)
    cov_ftv = lexical_coverage(ft, toks)
    print(f"  {f_path.name}: W2V={cov_w2v:.3f}, FT(vocab)={cov_ftv:.3f}")

if all_toks:
    print("  ---------------------------------")
    cov_w2v = lexical_coverage(w2v, all_toks)
    cov_ftv = lexical_coverage(ft, all_toks)
    print(f"  TOTAL: W2V={cov_w2v:.3f}, FT(vocab)={cov_ftv:.3f}")


# --- 2. Similarity ---
def cos(a, b): return float(dot(a, b) / (norm(a) * norm(b)))

def pair_sim(model, pairs):
    vals = []
    for a, b in pairs:
        try:
            vals.append(model.wv.similarity(a, b))
        except KeyError:
            pass
    return sum(vals) / len(vals) if vals else float('nan')

# Test pairs for synonyms and antonyms
syn_pairs = [("yaxşı", "əla"), ("bahalı", "qiymətli"), ("ucuz", "sərfəli")]
ant_pairs = [("yaxşı", "pis"), ("bahalı", "ucuz")]

syn_w2v = pair_sim(w2v, syn_pairs)
syn_ft = pair_sim(ft, syn_pairs)
ant_w2v = pair_sim(w2v, ant_pairs)
ant_ft = pair_sim(ft, ant_pairs)
sep_w2v = syn_w2v - ant_w2v
sep_ft = syn_ft - ant_ft

print("\n== 2. Similarity (Synonyms: high is good; Antonyms: low is good) ==")
print(f"  Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
print(f"  Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
print(f"  Separation (Syn-Ant): W2V={sep_w2v:.3f}, FT={sep_ft:.3f}")


# --- 3. Nearest Neighbors ---
def neighbors(model, word, k=5):
    try:
        return [w for w, score in model.wv.most_similar(word, topn=k)]
    except KeyError:
        return ["<WORD-NOT-FOUND>"]

seed_words = ["yaxşı", "pis", "çox", "bahalı", "ucuz", "mükəmməl", "dəhşət", "<PRICE>", "<RATING_POS>", "USER"]

print("\n== 3. Nearest Neighbors (Qualitative Analysis) ==")
for w in seed_words:
    print(f"  --- NN for '{w}' ---")
    print(f"    W2V: {neighbors(w2v, w)}")
    print(f"    FT : {neighbors(ft, w)}")

print("\nEvaluation complete.")