import tensorflow as tf
import numpy as np

from alice import build_alice
from bob import build_bob
from eve import build_eve
from utils import load_data

def bit_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred)

def evaluate(alice, bob, eve):
    messages = np.random.randint(0, 2, (1000, 16))
    keys = np.random.randint(0, 2, (1000, 16))

    # Convert to tensors
    messages_tf = tf.convert_to_tensor(messages, dtype=tf.float32)
    keys_tf = tf.convert_to_tensor(keys, dtype=tf.float32)

    c = alice([messages_tf, keys_tf])
    bob_pred = bob([c, keys_tf])
    # Add noise to Eve's input for evaluation
    c_noisy = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)
    eve_pred = eve(c_noisy)

    bob_acc = bit_accuracy(messages, bob_pred.numpy())
    eve_acc = bit_accuracy(messages, eve_pred.numpy())

    print("\nTask 4.1: Final Evaluation")
    print("Tested on 1,000 unseen messages/keys.")
    print(f"Bob's bit accuracy: {bob_acc:.2%}")
    print(f"Eve's bit accuracy: {eve_acc:.2%}")

    return bob_acc, eve_acc

if __name__ == "__main__":
    # Setup
    msg_len = 16
    epochs = 80  # Full training like original
    batch_size = 128

    alice = build_alice(msg_len)
    bob = build_bob(msg_len)
    eve = build_eve(msg_len)

    opt_ab = tf.keras.optimizers.Adam(0.001)
    opt_eve = tf.keras.optimizers.Adam(0.001)

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    messages, keys = load_data()

    dataset = tf.data.Dataset.from_tensor_slices((messages, keys))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Training functions
    @tf.function
    def train_eve(m, k):
        with tf.GradientTape() as tape:
            c = alice([m, k])
            c = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)
            eve_pred = eve(c)
            eve_loss = loss_fn(m, eve_pred)
        grads = tape.gradient(eve_loss, eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads, eve.trainable_variables))
        return eve_loss

    @tf.function
    def train_alice_bob(m, k):
        with tf.GradientTape() as tape:
            c = alice([m, k])
            bob_pred = bob([c, k])
            c_noisy = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)
            eve_pred = eve(c_noisy)
            bob_loss = loss_fn(m, bob_pred)
            eve_loss = loss_fn(m, eve_pred)
            total_loss = bob_loss + (1 - eve_loss)
        vars_ab = alice.trainable_variables + bob.trainable_variables
        grads = tape.gradient(total_loss, vars_ab)
        opt_ab.apply_gradients(zip(grads, vars_ab))
        return bob_loss, eve_loss

    print("Training models for final evaluation...")
    for epoch in range(epochs):
        for step, (m, k) in enumerate(dataset):
            m = tf.cast(m, tf.float32)
            k = tf.cast(k, tf.float32)

            if step % 2 == 0:
                train_eve(m, k)
            train_alice_bob(m, k)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")

    print("Training complete. Running final evaluation...")
    evaluate(alice, bob, eve)