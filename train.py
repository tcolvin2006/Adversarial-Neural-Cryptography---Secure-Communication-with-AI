import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from alice import build_alice
from bob import build_bob
from eve import build_eve
from utils import load_data

# ------------------------
# Setup
# ------------------------
msg_len = 16
epochs = 80
batch_size = 64

alice = build_alice(msg_len)
bob = build_bob(msg_len)
eve = build_eve(msg_len)

opt_ab = tf.keras.optimizers.Adam(0.001)
opt_eve = tf.keras.optimizers.Adam(0.001)

loss_fn = tf.keras.losses.BinaryCrossentropy()

messages, keys = load_data()

bob_accs = []
eve_accs = []

# ------------------------
# Accuracy function
# ------------------------
def bit_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred)

# ------------------------
# Training loop
# ------------------------
for epoch in range(epochs):
    for i in range(0, len(messages), batch_size):

        m = tf.convert_to_tensor(messages[i:i+batch_size], dtype=tf.float32)
        k = tf.convert_to_tensor(keys[i:i+batch_size], dtype=tf.float32)

        # ---------------- Eve training ----------------
        if i % 2 == 0:
            with tf.GradientTape() as tape:
                c = alice([m, k])
                c = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)

                eve_pred = eve(c)
                eve_loss = loss_fn(m, eve_pred)

            grads = tape.gradient(eve_loss, eve.trainable_variables)
            opt_eve.apply_gradients(zip(grads, eve.trainable_variables))

        # ---------------- Alice + Bob training ----------------
        with tf.GradientTape() as tape:
            c = alice([m, k])
            bob_pred = bob([c, k])
            eve_pred = eve(c)

            bob_loss = loss_fn(m, bob_pred)
            eve_loss = loss_fn(m, eve_pred)

            total_loss = bob_loss + (1 - eve_loss)

        vars_ab = alice.trainable_variables + bob.trainable_variables
        grads = tape.gradient(total_loss, vars_ab)
        opt_ab.apply_gradients(zip(grads, vars_ab))

    # ---------------- Evaluation per epoch ----------------
    m_eval = tf.convert_to_tensor(messages[:batch_size], dtype=tf.float32)
    k_eval = tf.convert_to_tensor(keys[:batch_size], dtype=tf.float32)

    c = alice([m_eval, k_eval])
    bob_pred = bob([c, k_eval])
    eve_pred = eve(c)

    bob_acc = bit_accuracy(messages[:batch_size], bob_pred.numpy())
    eve_acc = bit_accuracy(messages[:batch_size], eve_pred.numpy())

    bob_accs.append(bob_acc)
    eve_accs.append(eve_acc)

    print("epoch:", epoch + 1)
    print("bob loss:", bob_loss.numpy())
    print("eve loss:", eve_loss.numpy())
    print("bob acc:", bob_acc)
    print("eve acc:", eve_acc)
    print("------------------")

# ------------------------
# Plot results
# ------------------------
plt.plot(bob_accs, label="Bob Accuracy")
plt.plot(eve_accs, label="Eve Accuracy")
plt.legend()
plt.title("Training Accuracy Over Time")
plt.show()

# ------------------------
# Final evaluation (FIXED)
# ------------------------
messages_test = np.random.randint(0, 2, (1000, 16))
keys_test = np.random.randint(0, 2, (1000, 16))

# 🔥 FIX: convert BOTH to tensors BEFORE model use
messages_test = tf.convert_to_tensor(messages_test, dtype=tf.float32)
keys_test = tf.convert_to_tensor(keys_test, dtype=tf.float32)

c = alice([messages_test, keys_test])
bob_pred = bob([c, keys_test])
eve_pred = eve(c)

def bit_accuracy_final(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred)

print("\nFINAL RESULTS")
print("Bob Accuracy:", bit_accuracy_final(messages_test.numpy(), bob_pred.numpy()))
print("Eve Accuracy:", bit_accuracy_final(messages_test.numpy(), eve_pred.numpy()))