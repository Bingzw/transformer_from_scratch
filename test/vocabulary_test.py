from transformer_from_scratch.src.vocabulary import Vocabulary

import unittest
import torch


class TestVocabulary(unittest.TestCase):
    maxDiff = None

    def test_tokenize(self):
        input_sequence = "Hello my name is Bing and I am born with this name. I am learning genAI."
        output = Vocabulary([]).tokenize(input_sequence)
        self.assertEqual(
            [
                'BOS',
                'Hello',
                'my',
                'name',
                'is',
                'Bing',
                'and',
                'I',
                'am',
                'born',
                'with',
                'this',
                'name',
                '.',
                'I',
                'am',
                'learning',
                'genAI',
                '.',
                'EOS'],
            output
        )

    def test_init_vocab(self):
        input_sentences = ["Hello my name is Bing and I am born with this name. I am learning genAI."]
        vocab = Vocabulary(input_sentences)
        expected = {
            'BOS': 0,
            'EOS': 1,
            'PAD': 2,
            'Hello': 3,
            'my': 4,
            'name': 5,
            'is': 6,
            'Bing': 7,
            'and': 8,
            'I': 9,
            'am': 10,
            'born': 11,
            'with': 12,
            'this': 13,
            '.': 14,
            'learning': 15,
            'genAI': 16}

        self.assertEqual(vocab.token2index, expected)

    def test_encode(self):
        def test_encode(self):
            input_sentences = ["Hello my name is Bing and I am born with this name. I am learning genAI."]
            vocab = Vocabulary(input_sentences)
            output = vocab.encode(input_sentences[0])
            self.assertEqual(output, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 14, 9, 10, 15, 16, 14, 1])

        def test_encode_no_special_tokens(self):
            input_sentences = ["Hello my name is Bing and I am born with this name. I am learning genAI."]
            vocab = Vocabulary(input_sentences)
            output = vocab.encode(input_sentences[0], add_special_tokens=False)
            self.assertEqual(output, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 14, 9, 10, 15, 16, 14])

        def test_batch_encode(self):
            input_sentences = [
                "This is one sentence",
                "This is another, much longer sentence",
                "Short sentence",
            ]
            vocab = Vocabulary(input_sentences)
            output = vocab.batch_encode(input_sentences, add_special_tokens=False)
            self.assertEqual(
                output,
                [[3, 4, 5, 6, 2, 2, 2], [3, 4, 7, 8, 9, 10, 6], [11, 6, 2, 2, 2, 2, 2]],
            )


if __name__ == "__main__":
    unittest.main()