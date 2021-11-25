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

## Tokenizers and models
We use https://huggingface.co/dbmdz/bert-base-italian-cased as the italian tokenizer for each of our models, for the source language we must use the correct tokenizer for each encoder we used.
### Masked language encoders:
- https://huggingface.co/bert-base-uncased
- https://huggingface.co/distilbert-base-uncased
### Neural mahcine translation encoders:
- https://huggingface.co/t5-small
- https://huggingface.co/xlnet-base-cased
