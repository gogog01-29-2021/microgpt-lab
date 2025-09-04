# microgpt-lab

This repository serves as a playground for building and experimenting with minimal GPT‑like components. It is designed to be modular and extensible, allowing you to build up from simple pieces (such as tokenizers) to a full GPT stack.

The first module lives in `tokenizers/bpe_char_eow/` and implements a character‑level Byte Pair Encoding (BPE) tokenizer with an end‑of‑word marker (`</w>`). It includes:

- `train_bpe.py`: a CLI to train a BPE tokenizer on a corpus and write out `merges.txt` and `encoder.json`.
- `sample_corpus.txt`: a tiny example corpus.
- `merges.txt` and `encoder.json`: the learned merge rules and token‑to‑ID map from the sample corpus.
- `tests/test_roundtrip.py`: a simple unit test to verify encoding and decoding round‑trip.

As you extend this repository, consider adding additional tokenizers or other components (models, data loaders, training scripts) in sibling directories such as `tokenizers/bpe_byte_level/` or `models/`. Pull requests and suggestions are welcome.
