import unittest
from typing import List, Dict, Any
import random
from random import choices

import numpy as np
import torch
from torch import nn

from src.lr_scheduler import NoamOpt
from src.transformer import Transformer
from src.vocabulary import Vocabulary
from utils import construct_batches


def train(transformer: nn.Module, scheduler: Any, loss_criterion: Any, batches: Dict[str, List[torch.Tensor]],
          masks: Dict[str, List[torch.Tensor]], n_epochs: int):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param loss_criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)
    num_iters = 0

    for epoch in range(n_epochs):
        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
                zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            encoder_output = transformer.encoder(input_ids=src_batch, src_padding_mask=src_mask)
            # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(input_tokens=tgt_batch, encoder_hidden_states=encoder_output,
                                                 src_padding_mask=src_mask, future_mask=tgt_mask)
            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. The BOS token in the target is also not something we want to compute a loss for.
            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:]
            # Compute the loss
            batch_loss = loss_criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )

            batch_accuracy = (
                torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
            ) / tgt_batch.numel()

            if num_iters % 100 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {batch_loss.item()}, Accuracy: {batch_accuracy.item()}")

            # Backpropagation
            batch_loss.backward()
            scheduler.step()
            scheduler.optimizer.zero_grad()
            num_iters += 1

    return batch_loss, batch_accuracy


if __name__ == "__main__":
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if device.type == "cpu":
        print("This unit test was not run because it requires a GPU")

    # hyperparameters
    synthetic_corpus_size = 200
    batch_size = 20
    n_epochs = 200
    n_tokens_in_batch = 10

    # create synthetic corpus
    corpus = ["Generative artificial intelligence (generative AI, GenAI, or GAI) is artificial intelligence capable "
              "of generating text, images, videos, or other data using generative models, often in response to "
              "prompts."]
    print("original corpus: ", corpus)
    vocab = Vocabulary(corpus)
    vocab_size = len(vocab.token2index.items())
    valid_tokens = list(vocab.token2index.keys())[3:]
    print("valid tokens: ", valid_tokens)
    corpus += [
        " ".join(choices(valid_tokens, k=n_tokens_in_batch)) for _ in range(synthetic_corpus_size)
    ]
    print("synthetic corpus: ", corpus)
    print("len of synthetic corpus: ", len(corpus))

    # construct stc-tgt aligned input batches
    corpus = [{"src": sent, "tgt": sent} for sent in corpus]
    batches, masks = construct_batches(corpus, vocab, batch_size, "src", "tgt", device)
    print("source batch: ", batches['src'])
    print("number of batches: ", len(batches['src']))
    print("batch size: ", len(batches['src'][0]))
    print("target batch: ", batches['tgt'][0])
    print("masks: ", masks['tgt'][0])
    print("masks shape: ", masks['tgt'][0].shape)

    # create transformer model
    transformer = Transformer(
        hidden_dim=512,
        ff_dim=2048,
        num_heads=8,
        num_layers=2,
        max_decoding_length=25,
        vocab_size=vocab_size,
        padding_idx=vocab.token2index[vocab.PAD],
        bos_idx=vocab.token2index[vocab.BOS],
        dropout_p=0.1,
        tie_output_to_embedding=True,
    ).to(device)

    # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamOpt(
        transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
    )
    criterion = nn.CrossEntropyLoss()

    # Start training and verify ~zero loss and >90% accuracy on the last batch
    latest_batch_loss, latest_batch_accuracy = train(
        transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
    )





