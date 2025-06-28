# Semantic Search with Word2Vec Project

A modular, production-ready pipeline for building a semantic search engine using CBOW and SkipGram word embedding models trained on PDF document corpora.

## Features
- **Semantic Search:** Find and rank similar sentences or queries using vector space similarity (cosine similarity) over learned embeddings.
- **PDF Document Ingestion:** Reads and processes PDF files from a directory.
- **Text Preprocessing:** Cleans, tokenizes, and lemmatizes text using NLTK and Stanza.
- **Vocabulary Building:** Efficiently builds a vocabulary with frequency counts.
- **ID Mapping:** Maps words to unique IDs and vice versa.
- **Training Pipelines:**
  - CBOW and SkipGram model training scripts with configurable hyperparameters.
- **Embedding Generation:** Generates and saves sentence/corpus embeddings for fast search.
- **Configurable:** All paths and hyperparameters are managed in `config.py`.

## Project Structure
```
word2vec-model/
├── config.py                # Central config for paths and hyperparameters
├── data/                    # Model, vocab, and embedding files
├── docs/                    # Documentation and references
├── models/                  # Model definitions
├── scripts/                 # Training and utility scripts
│   ├── train_cbow.py
│   ├── train_skipgram.py
│   └── generate_corpus_embeddings.py
├── semantic_search/         # Semantic search logic
├── utils/                   # All utility modules
│   ├── document_reader.py
│   ├── preprocessor.py
│   ├── vocab_utils.py
│   ├── dataset_utils.py
│   ├── map_utils.py
│   ├── model_utils.py
│   └── train_pipeline.py
├── main.py                  # Entry point for semantic search
├── requirements.txt         # Python dependencies
└── README.md
```

## Quickstart
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure paths and hyperparameters:**
   Edit `config.py` as needed for your data and preferences.

3. **Prepare your data:**
   - Place your PDF files in the directory specified by `CORPUS_PATH` in `config.py`.

4. **Train a model:**
   - For CBOW:
     ```bash
     python scripts/train_cbow.py
     ```
   - For SkipGram:
     ```bash
     python scripts/train_skipgram.py
     ```

5. **Generate corpus embeddings:**
   ```bash
   python scripts/generate_corpus_embeddings.py
   ```

6. **Run semantic search:**
   ```bash
   python main.py
   ```
   Enter your query when prompted. The system will return the most semantically similar sentences from your corpus.

## Configuration
All paths and hyperparameters are set in `config.py`:
- `CORPUS_PATH`: Directory containing PDF files
- `VOCAB_PATH`, `MODEL_PATH`, `CORPUS_EMB_PATH`: Output files
- `EMBEDDING_DIM`, `WINDOW_SIZE`, `BATCH_SIZE`, `EPOCHS`: Model/training settings

## Requirements
- Python 3.8+
- See `requirements.txt` for Python packages

## Extending
- Add new preprocessing steps in `utils/text_preprocessor.py`.
- Add new models in `models/`.
- Add new scripts in `scripts/`.

## License
MIT License

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [NLTK](https://www.nltk.org/)
- [Stanza](https://stanfordnlp.github.io/stanza/)
- [PyPDF](https://pypdf.readthedocs.io/)
