import argparse
import json
import os
from collections import Counter


def get_vocab(corpus_path: str) -> Counter:
    """Build a vocabulary of words represented as tuples of characters with an end-of-word marker."""
    vocab = Counter()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                symbols = list(word) + ['</w>']
                vocab[tuple(symbols)] += 1
    return vocab


def get_stats(vocab: Counter) -> Counter:
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs


def merge_vocab(pair: tuple, vocab: Counter) -> Counter:
    new_vocab = Counter()
    bigram = ' '.join(pair)
    merged_symbol = ''.join(pair)
    for word, freq in vocab.items():
        word_str = ' '.join(word)
        new_word_str = word_str.replace(bigram, merged_symbol)
        new_word = tuple(new_word_str.split(' '))
        new_vocab[new_word] += freq
    return new_vocab


def bpe_train(corpus_path: str, vocab_size: int):
    vocab = get_vocab(corpus_path)
    tokens = set()
    for word in vocab:
        tokens.update(word)
    merges = []
    while len(tokens) < vocab_size:
        pair_stats = get_stats(vocab)
        if not pair_stats:
            break
        best_pair = pair_stats.most_common(1)[0][0]
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        tokens.add(''.join(best_pair))
    return tokens, merges


def assign_ids(tokens: set, merges: list) -> dict:
    merge_symbols = {''.join(m) for m in merges}
    base_tokens = sorted([tok for tok in tokens if tok not in merge_symbols])
    ordered_tokens = base_tokens + [''.join(m) for m in merges]
    return {tok: idx for idx, tok in enumerate(ordered_tokens)}


def write_files(merges: list, ids: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'merges.txt'), 'w', encoding='utf-8') as f:
        for pair in merges:
            f.write(f"{pair[0]} {pair[1]}\n")
    with open(os.path.join(out_dir, 'encoder.json'), 'w', encoding='utf-8') as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)


def apply_merges_to_word(symbols: list, merges: list) -> list:
    symbols = symbols.copy()
    for merge in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == merge[0] and symbols[i + 1] == merge[1]:
                symbols[i] = symbols[i] + symbols[i + 1]
                del symbols[i + 1]
            else:
                i += 1
    return symbols


def encode(words: list, merges: list, ids: dict) -> list:
    token_ids = []
    for word in words:
        symbols = list(word) + ['</w>']
        merged = apply_merges_to_word(symbols, merges)
        token_ids.extend([ids[s] for s in merged])
    return token_ids


def decode(token_ids: list, merges: list, ids: dict) -> str:
    inv = {v: k for k, v in ids.items()}
    words = []
    current = []
    for tid in token_ids:
        sym = inv[tid]
        if sym == '</w>':
            words.append(''.join(current))
            current = []
        elif sym.endswith('</w>'):
            # merged symbol that ends with </w>
            stripped = sym[:-4]
            current.append(stripped)
            words.append(''.join(current))
            current = []
        else:
            current.append(sym)
    if current:
        words.append(''.join(current))
    return ' '.join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    tokens, merges = bpe_train(args.corpus, args.vocab_size)
    ids = assign_ids(tokens, merges)
    write_files(merges, ids, args.out_dir)
    if args.demo:
        sample = 'the quick fox merges tokens greedily'
        encoded = encode(sample.split(), merges, ids)
        decoded = decode(encoded, merges, ids)
        print('Sample:', sample)
        print('Encoded:', encoded)
        print('Decoded:', decoded)


if __name__ == '__main__':
    main()
