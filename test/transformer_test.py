import random
import unittest
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from src.vocabulary import Vocabulary
from src.encoder import TransformerEncoder
from src.decoder import TransformerDecoder
from src.transformer import Transformer
from utils import construct_future_mask


class TestTransformer(unittest.TestCase):
    def test_transformer_inference(self):
        seed = 100
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        corpus = [
            "Hello my name is Bing and I am born with this name. I am learning genAI.",
            "genAI is artificial intelligence capable of generating text, images, videos, or other data using generative models, often in response to prompts"
        ]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        with torch.no_grad():
            transformer = Transformer(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=10,
                vocab_size=en_vocab_size,
                padding_idx=en_vocab.token2index[en_vocab.PAD],
                bos_idx=en_vocab.token2index[en_vocab.BOS],
                dropout_p=0.1,
                tie_output_to_embedding=True
            )
            transformer.eval()
            # prepare encode input, mask and generate output hidden states
            encoder_input = torch.IntTensor(
                en_vocab.batch_encode(corpus, add_special_tokens=False)
            )
            print("encoder_input", encoder_input)
            src_padding_mask = encoder_input != transformer.padding_idx
            encoder_output = transformer.encoder.forward(
                encoder_input, src_padding_mask=src_padding_mask
            )
            print("encoder_output", encoder_output)
            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)

            # prepare decoder input and mask and start decoding
            decoder_input = torch.IntTensor(
                [[transformer.bos_idx], [transformer.bos_idx]]
            )
            print("decoder_input", decoder_input)
            future_mask = construct_future_mask(seq_len=1)
            for i in range(transformer.max_decoding_length):
                decoder_output = transformer.decoder.forward(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask
                )
                print("decoder_output shape", decoder_output.shape)
                print(decoder_output[:, -1, :])
                # take the argmax over the softmax of the last token
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
                print("predicted_tokens", predicted_tokens)

                # append the prediction to decoded tokens and construct the new mask
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])

            self.assertEqual(decoder_input.shape, (2, transformer.max_decoding_length + 1))
            self.assertEqual(torch.all(decoder_input == transformer.bos_idx), False)


if __name__ == "__main__":
    unittest.main()
