# Character‑level BPE with end‑of‑word marker

This directory contains a minimal implementation of a character‑level Byte Pair Encoding (BPE) tokenizer. Each word is treated as a sequence of characters with an explicit end‑of‑word marker `</w>`. The trainer greedily merges the most frequent pair of adjacent symbols until a target vocabulary size is reached.

## Files

- **train_bpe.py** — Command‑line interface to train a BPE tokenizer. It reads a corpus, learns merge rules, assigns deterministic token IDs, writes `merges.txt` and `encoder.json`, and can optionally demonstrate a round‑trip encode/decode.
- **sample_corpus.txt** — A tiny corpus used for the demonstration and for unit testing.
- **merges.txt** — Merge rules in rank order, one pair per line.
- **encoder.json** — Mapping from token strings to integer IDs.
- **tests/test_roundtrip.py** — Unit test to ensure that encoding and decoding a sample sentence reproduces the original.

## Usage

To train on your own corpus and generate a vocabulary of a specific size, run:

```bash
python3 train_bpe.py --corpus path/to/corpus.txt --vocab_size 300 --out_dir /path/to/output
```

This will produce `merges.txt` and `encoder.json` in the specified output directory. Use them at inference time to encode raw text into token IDs and decode token IDs back into text.
