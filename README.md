# CENG 442 - Assignment 1: Azerbaijani Text Preprocessing + Word Embeddings

This repository was prepared for the CENG442 course, Assignment 1.

**Grup Members:**
* Burhan Koçak - 21050111044
* Emre Karaburu - 21050111064

---

### 1) Dataset and Objective

In this assignment, 5 different Azerbaijani Turkish datasets (labeled-sentiment, test_1_, etc.) were studied. The main objective is to clean these texts for sentiment analysis and to build a "domain-aware" text processing pipeline.

Sentiment values were converted into a three-class structure: Negative=0.0, Neutral=0.5, Positive=1.0. Preserving the neutral value as 0.5 allows the sentiment score to also be modeled as a regression problem and maintains the distance between the two extremes (0 and 1).

### 2) Preprocessing

Text cleaning was performed using the normalize_text_az function in the process_data.py script. The main rules applied are as follows:

* **Normalization:** Azerbaijani-specific transformations İ->i, I->ı and unicodedata.NFC normalization.
* **Cleaning:** HTML tags, URLs, emails, phone numbers, and @mentions were replaced with URL, EMAIL, PHONE, and USER tokens, respectively.
* **Numbers:** All digits were replaced with the <NUM> token.
* **Characters:** Repeated letters (e.g., cooool) were reduced to 2 (cool). Punctuation marks were removed.
* **Single Letters:** Single-letter tokens other than o and e were discarded.


**Example (Before/After):**

* **Before:** `Salam @user, bu #GozelFilmi cox beyendim 👍👍 https://film.az qiymeti 10 manat idi.`
* **After:** `salam USER gozel filmi çox bəyəndim EMO_POS URL qiyməti <NUM> manat idi`


### 3) Mini Challenges

* **Emoji Mapping:** Using the EMO_MAP dictionary, basic positive/negative emojis were labeled as EMO_POS and EMO_NEG.
* **Hashtag Split:** "camelCase" hashtags like #QarabagIsBack were split using re.sub to qarabag is back.
* **Negation Scope:** For 3 words following negation words like yox, deyil, heç, a _NEG suffix was added (e.g., yaxşı deyil -> yaxşı deyil_NEG). This ensures that the expression "not good" has a different vector than the word "good".
* **Simple Deasciify:** With the SLANG_MAP dictionary, common "deasciify" operations like cox -> çox and some abbreviations (tmm -> tamam) were corrected.
* **Stopword Research:** 
  - Purpose: Remove high-frequency function words that add noise while keeping sentiment-bearing tokens (especially negations and intensifiers).
  - Method: Start with corpus top-frequency terms plus standard lists; include conjunctions, prepositions, pronouns, auxiliaries with low sentiment value; exclude negations and intensifiers.
  - Negation exceptions: Do not remove AZ: “yox”, “deyil” (optional: “heç”); TR: “değil”, “yok”, “hiç”; EN: “not”, “no”, “never”.
  - Intensifiers: Keep “çox/çok/very” and similar.
  - Sample stopwords:
    - AZ: və, ilə, üçün, kimi, amma, lakin, çünki, həm, həm də, da, də, ki, bu, o, bir, hər, ən, artıq, sonra, əvvəl, belə, elə, həmçinin, bəzən, ya.
    - TR: ve, ile, için, gibi, ama, fakat, ancak, çünkü, hem, de, da, ki, bu, şu, o, bir, her, en, sonra, önce, ayrıca, yani, hatta, üzere.
    - EN: and, or, but, with, for, to, of, in, on, at, by, from, as, is, are, was, were, be, been, being, a, an, the, this, that, these, those, which, who.
  - Notes: If tokens carry a _NEG suffix from negation scope tagging, do not remove them. If clitics “da/de” are attached as suffixes, do not strip them as stopwords; only remove when they are standalone tokens.

### 4) Domain-Aware Approaches

Texts were divided into four classes using simple rules: `news`, `social`, `reviews`, `general`.

- Detection (`detect_domain`): If the text contains `apa` or `trend`, label as "news"; if it contains `@` or `#`, label as "social"; if it contains `azn`, `manat`, or `ulduz`, label as "reviews".
- Normalization (`domain_specific_normalize`): For the "reviews" domain, apply special transforms such as `10 azn` -> `<PRICE>` or `5 ulduz` -> `<STARS_5>`.
- Corpus Tagging: Prefix each line in `corpus_all.txt` with domain tags like `domnews`, `domsocial`, etc.

### 5) Embeddings

Word2Vec (Skip-Gram) and FastText (Skip-Gram) models were trained using the `gensim` library.

**Training Settings:**
| Parameter | Word2Vec | FastText |
| :--- | :--- | :--- |
| `vector_size` | 300 | 300 |
| `window` | 5 | 5 |
| `min_count` | 3 | 3 |
| `sg` | 1 (Skip-Gram) | 1 (Skip-Gram) |
| `epochs` | 10 | 10 |
| `min_n / max_n`| - | 3 / 6 |

**Processed Data Files:**
The five two-column Excel output files and the corpus_all.txt file are in the output folder.

**Model Files (Embeddings):**
The files `word2vec.model` and `fasttext.model`, which should be located in the `embeddings/` folder, have been made accessible via an external Google Drive link because they exceed the GitHub file size limits (2.6 GB).

* **Models Download Link:** `https://drive.google.com/drive/folders/1etLwTxWDvdoc9QvkEWpevQTs-j6Hi04i?usp=sharing`

**Evaluation Results - Terminal Output (from evaluate_models.py):**

--- Step 4: Evaluating embedding models ---
Models loaded successfully.

== 1. Lexical Coverage (per dataset) ==
  labeled-sentiment_2col.xlsx: W2V=0.929, FT(vocab)=0.929
  test__1__2col.xlsx: W2V=0.984, FT(vocab)=0.984
  train__3__2col.xlsx: W2V=0.987, FT(vocab)=0.987
  train-00000-of-00001_2col.xlsx: W2V=0.934, FT(vocab)=0.934
  merged_dataset_CSV__1__2col.xlsx: W2V=0.941, FT(vocab)=0.941
  ---------------------------------
  TOTAL: W2V=0.948, FT(vocab)=0.948

== 2. Similarity (Eşanlamlı: yüksək iyi; Zıtanlamlı: düşük iyi) ==
  Eşanlamlılar: W2V=0.357, FT=0.448
  Zıtanlamlılar: W2V=0.340, FT=0.419
  Ayrışma (Syn-Ant): W2V=0.017, FT=0.029

== 3. Nearest Neighbors (Kalitatif Analiz) ==
  --- NN for 'yaxşı' ---
    W2V: ['<RATING_POS>', 'iyi', 'yaxshi', 'yaxwi', 'yaxsı']
    FT : ['yaxşıyaxşı', 'yaxşıı', 'yaxşıkı', 'yaxşıca', 'yaxş']
  --- NN for 'pis' ---
    W2V: ['vərdişlərə', 'gündә', 'sürükliyirtort', 'millәt', '<RATING_NEG>']
    FT : ['pisbu', 'piis', 'pispul', 'pi', 'çoxpis']
  --- NN for 'çox' ---
    W2V: ['gözəldir', 'çöx', 'əladir', 'çoox', 'bəyənirəm']
    FT : ['sçox', 'çoxx', 'çoxçox', 'çoxh', 'azçox']
  --- NN for 'bahalı' ---
    W2V: ['metallarla', 'radiusda', 'portretlerinə', 'içməklər', 'yaxtaları']
    FT : ['bahalıı', 'bahalısı', 'bahalıq', 'baharlı', 'bahalığı']
  --- NN for 'ucuz' ---
    W2V: ['düzəltdirilib', 'qiymete', 'qiymətə', 'keyfiyetli', 'sududu']
    FT : ['ucuza', 'ucuzu', 'ucuzhəm', 'ucuzdu', 'ucu']
  --- NN for 'mükəmməl' ---
    W2V: ['möhtəşəmm', 'bayıldım', 'kəliməylə', 'süjetli', 'tamamlayır']
    FT : ['mükəmməll', 'mükəməl', 'mukəmməl', 'mükəmməldi', 'mükəmməlsiz']
  --- NN for 'dəhşət' ---
    W2V: ['xalçalardan', 'ayranları', 'açdıq', 'birda', 'təsirlidi']
    FT : ['dəhşətdü', 'dəhşətə', 'dəhşəti', 'dəhşətizm', 'dəhşətdi']
  --- NN for '<PRICE>' ---
    W2V: ['<KELIME-YOK>']
    FT : ['engiltdere', 'recebzade', 'esgere', 'felestine', 'fatihe']
  --- NN for '<RATING_POS>' ---
    W2V: ['deneyin', 'uygulama', 'süper', 'hak', 'bence']
    FT : ['<RATING_NEG>', 'süperr', 'süper', 'çookk', 'cöx']
  --- NN for 'USER' ---
    W2V: ['<KELIME-YOK>']
    FT : ['qərənfillər', 'qərəzli', 'aktyorluq', 'emosional', 'əsaslanırmış']

Evaluation complete.