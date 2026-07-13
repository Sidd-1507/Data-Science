# GPT From Scratch

A character-level, decoder-only Transformer (GPT) built entirely from scratch in PyTorch — implemented, trained, and explained step by step in a single Jupyter notebook.

Inspired by Andrej Karpathy's *"Let's build GPT: from scratch, in code, spelled out"*, this project re-derives and re-explains every component (tokenization, batching, self-attention, multi-head attention, residual connections, LayerNorm) so it can be followed end-to-end by anyone with basic Python and a little linear algebra/calculus — not just copied.

It goes beyond the base lecture with:
- **Loading real pretrained GPT-2 (124M) weights** into the same architecture, to prove correctness by generating fluent English, not just Shakespeare-flavored text.
- A **real, working supervised fine-tuning (SFT) pass** on a small instruction dataset — the actual mechanism (at small scale) that takes a model from "document completer" toward "instruction-following assistant," with a before/after comparison.
- A working, **numerically-tested implementation of Rotary Position Embeddings (RoPE)**.
- A **multilingual tokenization** demo.
- An honest, concrete walkthrough of what's still *not* included at full scale — the reward-modeling / RLHF stages that get a model the rest of the way to something like ChatGPT.

## What's inside

| # | Section | What it covers |
|---|---|---|
| 1 | Dataset & tokenization | Tiny Shakespeare + character-level tokenizer |
| 2 | Batching | `block_size`, `batch_size`, and why chunking is necessary |
| 3 | Bigram baseline | Simplest possible language model, as a sanity check |
| 4 | The attention math trick | Weighted averaging via triangular matrix multiplication |
| 5 | Self-attention | Query / key / value, scaled dot-product attention |
| 6 | Multi-head attention | Running several attention heads in parallel |
| 7 | Feed-forward network | Per-token computation after communication |
| 8 | Transformer block | Residual connections + pre-norm LayerNorm |
| 9 | Full GPT model | Token + position embeddings, stacked blocks, LM head |
| 10 | Training | The training loop, loss curves |
| 11 | Generation | Autoregressive sampling |
| 12 | Scaling to GPT-2 (124M) | Exact hyperparameter mapping to the real GPT-2 config |
| 13 | Loading real GPT-2 weights | Fluent-English generation from OpenAI's actual checkpoint |
| 14 | Bonus: RoPE | Rotary position embeddings, with a correctness test |
| 15 | Bonus: multilingual data | Tokenizing mixed-language text |
| 16 | What's not included | SFT, reward modeling, and RLHF — explained honestly |
| 17 | Real SFT pass | An actual working fine-tuning run that makes GPT-2 follow instructions |

## Getting started

### Option A: Google Colab (recommended)
Upload `GPT_from_scratch.ipynb` to [Google Colab](https://colab.research.google.com/), set the runtime to a GPU (`Runtime > Change runtime type > T4 GPU`), and run all cells top to bottom.

### Option B: Local / Jupyter
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook GPT_from_scratch.ipynb
```

A GPU is strongly recommended for Section 10 (training) at the default hyperparameters. If you're on CPU only, shrink `n_embd`, `n_layer`, `n_head`, `block_size`, and `max_iters` as noted in that section's markdown.

Section 13 (loading real GPT-2 weights) additionally requires internet access to download the checkpoint via Hugging Face (`transformers`).

## Model size / results

The default configuration trains a ~10M parameter model on ~1M characters of Shakespeare, reaching a validation loss of roughly **1.4–1.5** in a few thousand steps on a single GPU (a few minutes) — down from ~2.5 for the bigram baseline. Output is Shakespeare-*flavored* (character names, dialogue structure, archaic vocabulary) but not semantically coherent, which is expected at this scale; the point of the project is the architecture and mechanics, not competing with production LLMs.

## Acknowledgments

Structure and pedagogy inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his "Let's build GPT" lecture. The Tiny Shakespeare dataset is from his [char-rnn](https://github.com/karpathy/char-rnn) repository.

## License

MIT — see [LICENSE](LICENSE).
