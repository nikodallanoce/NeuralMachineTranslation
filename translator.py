import tensorflow as tf
from transformers import BertTokenizer


class Translator(tf.Module):
    def __init__(self, src_tokenizer: BertTokenizer, targ_tokenizer: BertTokenizer, transformer):
        super(Translator, self).__init__()
        self.src_tokenizer = src_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.transformer = transformer

    def __call__(self, sentence, max_length=30):
        # input sentence is portuguese, hence adding the start and end token
        # assert isinstance(sentence, tf.Tensor)
        # if len(sentence.shape) == 0:
        #  sentence = sentence[tf.newaxis]

        sentence_tok = self.src_tokenizer(sentence, padding='max_length', return_tensors='tf', max_length=max_length,
                                          add_special_tokens=True).data['input_ids']
        # sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = tf.reshape(sentence_tok, [1, max_length])
        print(encoder_input)

        # as the target is english, the first token to the transformer should be the
        # english start token.
        # start_end = self.tokenizers.en.tokenize([''])[0]
        # start = start_end[0][tf.newaxis]
        # end = start_end[1][tf.newaxis]

        start_end = self.targ_tokenizer("", padding='max_length', return_tensors='np',
                                        max_length=3, add_special_tokens=True).data['input_ids']

        start, end = tf.convert_to_tensor([start_end[0, 0]], dtype=tf.int32), tf.convert_to_tensor([start_end[0, 1]],
                                                                                                   dtype=tf.int32)

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            # print(predictions)

            predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())

        # output.shape (1, tokens)
        # text = tokenizers.en.detokenize(output)[0]  # shape: ()
        print(output)
        o = []
        for r in output:
            for e in r:
                o.append(e)
        text = self.targ_tokenizer.convert_ids_to_tokens(o)

        # tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([encoder_input, output[:, :-1]], training=False)

        return self.targ_tokenizer.convert_tokens_to_string(text), attention_weights
