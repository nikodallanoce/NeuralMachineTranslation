from transformers import BertTokenizer, TFBertModel

from DecoderLayer import DecoderLayer
import tensorflow as tf

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: TFBertModel = TFBertModel.from_pretrained("bert-base-uncased", trainable=True)
    tokens = tokenizer("good job guys", add_special_tokens=True, truncation=True, padding="max_length",
                       return_attention_mask=True, return_token_type_ids=True, return_tensors="tf", max_length=25)
    ret = model.call(**tokens, output_attentions=True, output_hidden_states=True)

    sample_decoder_layer = DecoderLayer(768, 8, 2048)
    sample_encoder_layer_output = ret.last_hidden_state

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((1, 50, 768)), sample_encoder_layer_output,
        False, None, None)

    sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)
