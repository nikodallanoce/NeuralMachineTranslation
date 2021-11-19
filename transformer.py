import tensorflow as tf
from tensorflow.keras import layers
from encoder import EncoderTransformer
from decoder import DecoderTransformer
from transformers import BertTokenizer, BertTokenizerFast, TFBertModel,\
    T5TokenizerFast, T5Tokenizer, TFT5EncoderModel,\
    DistilBertTokenizer, DistilBertTokenizerFast, TFDistilBertModel,\
    XLNetTokenizerFast, XLNetTokenizer, TFXLNetModel,\
    AlbertTokenizer, AlbertTokenizerFast, TFAlbertModel


def choose_strategy():
    # Detect hardware
    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        gpus = 0
    except ValueError:
        tpu_resolver = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu_resolver:
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        strategy = tf.distribute.TPUStrategy(tpu_resolver)
        print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on CPU')

    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy


def create_encoder_bert(strategy):
    with strategy.scope():
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    return bert_model


def create_encoder_t5(strategy):
    with strategy.scope():
        t5_model = TFT5EncoderModel.from_pretrained("t5-base"),
    return t5_model


def create_encoder_distilbert(strategy):
    with strategy.scope():
        distilbert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased"),
    return distilbert_model


def create_encoder_albert(strategy):
    with strategy.scope():
        albert_model = TFAlbertModel.from_pretrained("albert-base-v2"),
    return albert_model


def create_encoder_xlnet(strategy):
    with strategy.scope():
        xlnet_model = TFXLNetModel.from_pretrained("xlnet-base-cased"),
    return xlnet_model


encoder_models = {
    "bert": {
        "tokenizer": BertTokenizerFast.from_pretrained("bert-base-uncased"),
        "encoder": create_encoder_bert,
        "tokenizer_translation": BertTokenizer.from_pretrained("bert-base-uncased"),
    },
    "t5": {
        "tokenizer": T5TokenizerFast.from_pretrained("t5-base"),
        "encoder": create_encoder_t5,
        "tokenizer_translation": T5Tokenizer.from_pretrained("t5-base"),
    },
    "distil_bert": {
        "tokenizer": DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
        "encoder": create_encoder_distilbert,
        "tokenizer_translation": DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
    },
    "albert": {
        "tokenizer": AlbertTokenizerFast.from_pretrained("distilbert-base-uncased"),
        "encoder": create_encoder_albert,
        "tokenizer_translation": AlbertTokenizer.from_pretrained("distilbert-base-uncased"),
    },
    "xlnet": {
        "tokenizer": XLNetTokenizerFast.from_pretrained("xlnet-base-cased"),
        "encoder": create_encoder_xlnet,
        "tokenizer_translation": XLNetTokenizer.from_pretrained("xlnet-base-cased"),
    },
}


def create_model(layers_size: int,
                 num_layers: int,
                 dense_size: int,
                 num_heads: int,
                 max_length: int,
                 v_size_src: int,
                 v_size_dst: int,
                 encoder=None) -> tf.keras.Model:
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    if encoder is not None:
        outputs = encoder(encoder_inputs)
        encoder_outputs = outputs.last_hidden_state
        layers_size = encoder_outputs.shape[-1]  # the size of the encoder and decoder layers must be the same
    else:
        encoder_outputs = EncoderTransformer(num_layers, layers_size, dense_size, num_heads,
                                             max_length, v_size_src)(encoder_inputs)

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = tf.keras.Input(shape=(None, layers_size), name="decoder_state_inputs")
    decoder_outputs = DecoderTransformer(num_layers, layers_size, dense_size, num_heads,
                                         max_length, v_size_dst)(decoder_inputs, encoded_seq_inputs)
    decoder_outputs = layers.Dense(v_size_dst, activation="softmax")(decoder_outputs)
    decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs, name="decoder_transformer")

    # Final model
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
    return transformer
