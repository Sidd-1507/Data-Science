# 📘 Text Generation with RNN, LSTM & GRU — Learning Project

A beginner-friendly deep learning project comparing **Vanilla RNN, LSTM, and GRU**
architectures for next-word prediction and text generation, built with TensorFlow/Keras.

## 🎯 Goal

Train the same text corpus on three different sequence model architectures and
understand, both qualitatively and quantitatively, why gated architectures (LSTM, GRU)
tend to outperform a vanilla RNN on sequence modeling tasks.

## 📂 Repository Structure

```
.
├── notebooks/
│   ├── Text_Generation_RNN_LSTM_GRU_Learning_Project.ipynb           # Core version
│   └── Text_Generation_RNN_LSTM_GRU_Learning_Project_Detailed.ipynb  # Expanded version
├── requirements.txt
├── LICENSE
└── README.md
```

### Core notebook
The original learning project: tokenization, n-gram sequence creation, three models
(SimpleRNN, LSTM, GRU), a training loss/accuracy comparison, and a simple greedy
text generation function.

### Detailed notebook
Everything in the core notebook, plus:
- Corpus exploration (word frequency) and an optional larger sample corpus
- `model.summary()` and trainable parameter counts for all three architectures
- Training time tracking and `EarlyStopping`
- **Perplexity** calculation (the standard language-model evaluation metric)
- A side-by-side `pandas` comparison table (parameters, loss, accuracy, training time)
- **Temperature-based sampling** for text generation, compared against greedy decoding
- PCA visualization of each model's learned word embeddings
- Top-5 next-word probability inspection per model
- Model saving (`.keras` format)
- Tiered student tasks (beginner / intermediate / advanced)

## 🧠 Models Compared

| Model     | Gate Mechanism                  | Relative Parameters | Notes                                   |
|-----------|----------------------------------|----------------------|------------------------------------------|
| SimpleRNN | None                              | 1x (baseline)        | Struggles with long-term dependencies (vanishing gradients) |
| LSTM      | Input, forget, output gates       | ~4x                   | Strong long-range memory, more parameters |
| GRU       | Reset, update gates                | ~3x                   | Similar performance to LSTM, fewer parameters, often faster |

## 🚀 Getting Started

### Option 1: Open in Google Colab
Upload either notebook from the `notebooks/` folder directly to
[Google Colab](https://colab.research.google.com/) via **File → Upload notebook**.
No local setup required — Colab already has TensorFlow installed.

### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/Text_Generation_RNN_LSTM_GRU_Learning_Project_Detailed.ipynb
```

## 📦 Requirements

See [`requirements.txt`](requirements.txt). Core dependencies:
- TensorFlow
- NumPy
- Matplotlib
- pandas
- scikit-learn (detailed notebook only — used for PCA embedding visualization)

## 📚 Student Learning Tasks

The notebooks include tiered exercises:
- **Beginner** — swap in your own corpus, tweak embedding size and hidden units
- **Intermediate** — try the extended corpus, bidirectional layers, validation splits
- **Advanced** — stacked recurrent layers, character-level tokenization, beam search decoding

## ⚠️ A Note on the Sample Corpus

The built-in corpus is intentionally tiny (6–8 short lines) so the notebooks run in
seconds. This means all three models will tend to memorize it quickly and show very
low loss / perplexity — that's expected, not a bug, and is itself a useful illustration
of overfitting on small datasets. Swap in a larger corpus (a public-domain book, song
lyrics, etc.) to see more realistic behavior.

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
