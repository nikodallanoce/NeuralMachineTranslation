from transformer import *
from utilities import *
import time


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


class Trainer:

    def __init__(self, layers_size: int, transformer: TransformerNMT):
        learning_rate = CustomSchedule(layers_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.transformer = transformer

    @tf.function()
    def __train_step(self, src: tf.Tensor, dst: tf.Tensor) -> None:
        dst_inp = dst[:, :-1]
        dst_real = dst[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer.call([src, dst_inp], training=True)
            loss = loss_function(dst_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(dst_real, predictions))

    def train(self, epochs: int, tr_batches) -> None:
        for epoch in range(epochs):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (src, dst)) in enumerate(tr_batches):
                self.__train_step(src, dst)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} '
                        f'Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            if (epoch + 1) % 5 == 0:
                print("save")

            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


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
