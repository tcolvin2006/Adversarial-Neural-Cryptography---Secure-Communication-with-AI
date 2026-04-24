import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from alice import build_alice
from bob import build_bob
from eve import build_eve
from utils import load_data

# ------------------------
# Setup
# ------------------------
msg_len = 16
epochs = 80
batch_size = 128  # Increased batch size for better GPU utilization

alice = build_alice(msg_len)
bob = build_bob(msg_len)
eve = build_eve(msg_len)

opt_ab = tf.keras.optimizers.Adam(0.001)
opt_eve = tf.keras.optimizers.Adam(0.001)

loss_fn = tf.keras.losses.BinaryCrossentropy()

messages, keys = load_data()

# Create dataset for efficient batching and prefetching
dataset = tf.data.Dataset.from_tensor_slices((messages, keys))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Fixed evaluation data
eval_messages = messages[:batch_size]
eval_keys = keys[:batch_size]

bob_accs = []
eve_accs = []
bob_losses = []
eve_losses = []
# Accuracy function
# ------------------------
def bit_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred)

# ------------------------
# Training functions (compiled for speed)
# ------------------------
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
        # Add noise to Eve's input
        c_noisy = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)
        eve_pred = eve(c_noisy)
        bob_loss = loss_fn(m, bob_pred)
        eve_loss = loss_fn(m, eve_pred)
        total_loss = bob_loss + (1 - eve_loss)
    vars_ab = alice.trainable_variables + bob.trainable_variables
    grads = tape.gradient(total_loss, vars_ab)
    opt_ab.apply_gradients(zip(grads, vars_ab))
    return bob_loss, eve_loss

# ------------------------
# Training loop
# ------------------------
for epoch in range(epochs):
    step = 0
    epoch_bob_loss = 0
    epoch_eve_loss = 0
    num_steps = 0
    for m, k in dataset:
        # Convert to float32 if needed (mixed precision handles it)
        m = tf.cast(m, tf.float32)
        k = tf.cast(k, tf.float32)

        # ---------------- Eve training ----------------
        if step % 2 == 0:
            eve_loss = train_eve(m, k)
            epoch_eve_loss += eve_loss.numpy()

        # ---------------- Alice + Bob training ----------------
        bob_loss, eve_loss_ab = train_alice_bob(m, k)
        epoch_bob_loss += bob_loss.numpy()
        epoch_eve_loss += eve_loss_ab.numpy()
        step += 1
        num_steps += 1

    # Average losses for the epoch
    avg_bob_loss = epoch_bob_loss / num_steps
    avg_eve_loss = epoch_eve_loss / num_steps

    # ---------------- Evaluation per epoch ----------------
    m_eval = tf.convert_to_tensor(eval_messages, dtype=tf.float32)
    k_eval = tf.convert_to_tensor(eval_keys, dtype=tf.float32)

    c = alice([m_eval, k_eval])
    bob_pred = bob([c, k_eval])
    # Add noise to Eve's input for evaluation
    c_noisy = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)
    eve_pred = eve(c_noisy)

    bob_acc = bit_accuracy(eval_messages, bob_pred.numpy())
    eve_acc = bit_accuracy(eval_messages, eve_pred.numpy())

    bob_accs.append(bob_acc)
    eve_accs.append(eve_acc)
    bob_losses.append(avg_bob_loss)
    eve_losses.append(avg_eve_loss)

    print("epoch:", epoch + 1)
    print("bob loss:", avg_bob_loss)
    print("eve loss:", avg_eve_loss)
    print("bob acc:", bob_acc)
    print("eve acc:", eve_acc)
    print("------------------")

# ------------------------
# Plot results
# ------------------------
plt.figure(figsize=(10, 6))
plt.plot(bob_losses, label="Bob Training Loss", color='green', linewidth=2)
plt.plot(eve_losses, label="Eve Training Loss", color='orange', linewidth=2)
plt.legend()
plt.title("Training Loss Curves Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.savefig("training_loss_curves.png", dpi=300, bbox_inches='tight')
print("Training loss curves saved as training_loss_curves.png")
# plt.show()  # Commented out for headless environment

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
# Add noise to Eve's input for final evaluation
c_noisy = c + tf.random.normal(shape=tf.shape(c), stddev=0.1)
eve_pred = eve(c_noisy)

def bit_accuracy_final(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred)

print("\nFINAL RESULTS")
print("Bob Accuracy:", bit_accuracy_final(messages_test.numpy(), bob_pred.numpy()))
print("Eve Accuracy:", bit_accuracy_final(messages_test.numpy(), eve_pred.numpy()))