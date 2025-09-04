import unittest
from train_bpe import bpe_train, assign_ids, encode, decode


class TestBPERoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        # Train a small tokenizer on the sample corpus
        tokens, merges = bpe_train('sample_corpus.txt', vocab_size=200)
        ids = assign_ids(tokens, merges)
        sample = 'the quick fox merges tokens greedily'
        encoded = encode(sample.split(), merges, ids)
        decoded = decode(encoded, merges, ids)
        self.assertEqual(decoded, sample)


if __name__ == '__main__':
    unittest.main()
