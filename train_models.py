# train_models.py
# Trained Word2Vec and FastText embedding models on Azerbaijani sentiment datasets

import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec, FastText
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("--- Step 3: Training Word2Vec and FastText models ---")

# Processed dataset files from step 1
files = [
    "output/labeled-sentiment_2col.xlsx",
    "output/test__1__2col.xlsx",
    "output/train__3__2col.xlsx",
    "output/train-00000-of-00001_2col.xlsx",
    "output/merged_dataset_CSV__1__2col.xlsx",
]

sentences = []
print("Reading cleaned data from 2-column Excel files...")
for f in files:
    try:
        df = pd.read_excel(f, usecols=["cleaned_text"])
        sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())
    except Exception as e:
        print(f"Could not read {f}. Error: {e}")

print(f"Total sentences read for training: {len(sentences)}")
if not sentences:
    print("No sentences found. Exiting.")
    exit()

Path("embeddings").mkdir(exist_ok=True)

# --- Word2Vec Training ---
# Used skip-gram model with negative sampling
print("\nTraining Word2Vec model...")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,
    negative=10,
    epochs=10,
    workers=4
)
w2v_model.save("embeddings/word2vec.model")
print("Word2Vec model saved to 'embeddings/word2vec.model'")

# --- FastText Training ---
# Used skip-gram with character n-grams for OOV handling
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
    workers=4
)
ft_model.save("embeddings/fasttext.model")
print("FastText model saved to 'embeddings/fasttext.model'")

print("\nModel training complete.")