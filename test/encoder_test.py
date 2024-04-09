from src.encoder import TransformerEncoder
from src.vocabulary import Vocabulary
import unittest
import torch


class TestTransformerEncoder(unittest.TestCase):
    def test_transformer_encoder_single_sequence_batch(self):
        batch = ["Hello my name is Bing and I am born with this name. I am learning genAI."]
        en_vocab = Vocabulary(batch)
        en_vocab_size = len(en_vocab.token2index.items())
        with torch.no_grad():
            encoder = TransformerEncoder(
                embedding=torch.nn.Embedding(en_vocab_size, 512),
                hidden_dim=512,
                ff_dim=128,
                num_heads=8,
                num_layers=2,
                dropout_p=0.1
            )
            encoder._reset_parameters()
            encoder.eval()
            input_batch = torch.IntTensor(
                en_vocab.batch_encode(batch, add_special_tokens=False)
            )

            output = encoder.forward(input_batch)
            self.assertEqual(output.shape, (1, 18, 512))
            self.assertEqual(torch.any(torch.isnan(output)), False)

    def test_transformer_encoder_multi_sequence_batch(self):
        batch = [
            "Hello my name is Bing and I am born with this name. I am learning genAI.",
            "A shorter sequence in the batch"
        ]
        en_vocab = Vocabulary(batch)
        en_vocab_size = len(en_vocab.token2index.items())

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        with torch.no_grad():
            encoder = TransformerEncoder(
                embedding=torch.nn.Embedding(en_vocab_size, 6),
                hidden_dim=6,
                ff_dim=3,
                num_heads=2,
                num_layers=1,
                dropout_p=0.1,
            )
            encoder.eval()
            input_batch = torch.IntTensor(
                en_vocab.batch_encode(batch, add_special_tokens=False, padding=True)
            )
            src_padding_mask = input_batch != en_vocab.token2index[en_vocab.PAD]
            output = encoder.forward(input_batch, src_padding_mask=src_padding_mask)
            self.assertEqual(output.shape, (2, 18, 6))
            self.assertEqual(torch.any(torch.isnan(output)), False)


if __name__ == "__main__":
    unittest.main()