# HLT
Human Language Technologies project for the a.y. 2020/2021.
## Neural Machine Translation task
English-Italian

Europarl Corpus or the english-italian dataset from http://www.manythings.org/anki/

## Sources used
- https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
- https://www.tensorflow.org/text/tutorials/transformer
- https://keras.io/examples/nlp/neural_machine_translation_with_transformer/

## Benchmark
- https://github.com/facebookresearch/flores/blob/main/README.md

The baselines for confronting the results of our models were chosen from:
- MarianMT, https://huggingface.co/Helsinki-NLP/opus-mt-en-it
- DeltaLM, https://arxiv.org/pdf/2106.13736.pdf

## Tokenizers and models
We use https://huggingface.co/dbmdz/bert-base-italian-cased as the italian tokenizer for each of our models, for the source language we must use the correct tokenizer for each encoder we used.
### Masked language encoders:
- https://huggingface.co/distilbert-base-cased
### Neural machine translation encoders:
- https://huggingface.co/google/t5-v1_1-small
- https://huggingface.co/xlnet-base-cased
