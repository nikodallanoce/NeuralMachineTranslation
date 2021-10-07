from transformer import *


class Trainer:

    def __int__(self, layers_size, transformer: TransformerNMT):
        learning_rate = CustomSchedule(layers_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.transformer = transformer

    """@tf.function()
    def train_step(self, src, dst):
        tar_inp = dst[:, :-1]
        tar_real = dst[:, 1:]

        def step_fn(tar_inp, tar_real):
            with tf.GradientTape() as tape:
                predictions, _ = self.transformer.call([src, tar_inp], training=True)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, self.transformer.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))

        strategy.run(step_fn, args=(tar_inp, tar_real))"""


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
