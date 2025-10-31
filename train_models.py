# train_models.py
# Bu script, CENG442 Ödev 1'in 8. bölümündeki kodları kullanır.
# 

import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec, FastText
import logging

# Gensim loglamasını açarak ilerlemeyi görün
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("--- Step 3: Training Word2Vec and FastText models ---")

# Adım 1'de üretilen dosyaların yolları
files = [
    "output/labeled-sentiment_2col.xlsx",
    "output/test__1__2col.xlsx",
    "output/train__3__2col.xlsx",
    "output/train-00000-of-00001_2col.xlsx",
    "output/merged_dataset_CSV__1__2col.xlsx",
] #  (Dosya adlarını kendi çıktınıza göre güncelleyin)

sentences = []
print("Reading cleaned data from 2-column Excel files...")
for f in files:
    try:
        # Sadece temizlenmiş metin sütununu oku [cite: 305]
        df = pd.read_excel(f, usecols=["cleaned_text"])
        # Cümleleri token listelerine böl [cite: 306]
        sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())
    except Exception as e:
        print(f"Could not read {f}. Error: {e}")

print(f"Total sentences read for training: {len(sentences)}")
if not sentences:
    print("No sentences found. Exiting.")
    exit()

# Model çıktısı için klasör oluştur [cite: 307]
Path("embeddings").mkdir(exist_ok=True)

# --- Word2Vec Eğitimi ---
# Parametreler PDF'ten alındı [cite: 308-309]
# sg=1 -> skip-gram modeli
print("\nTraining Word2Vec model...")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,
    negative=10,
    epochs=10,
    workers=4 # Eğitimi hızlandırmak için
)
w2v_model.save("embeddings/word2vec.model") # [cite: 310]
print("Word2Vec model saved to 'embeddings/word2vec.model'")

# --- FastText Eğitimi ---
# Parametreler PDF'ten alındı [cite: 311-313]
# sg=1 -> skip-gram modeli
# min_n=3, max_n=6 -> karakter n-gramları (OOV kelimeler için)
print("\nTraining FastText model...")
ft_model = FastText(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,
    min_n=3,
    max_n=6,
    epochs=10,
    workers=4 # Eğitimi hızlandırmak için
)
ft_model.save("embeddings/fasttext.model") # [cite: 314]
print("FastText model saved to 'embeddings/fasttext.model'")

print("\nModel training complete. [cite: 315]")