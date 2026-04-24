import tensorflow as tf
import numpy as np

from alice import build_alice
from bob import build_bob
from eve import build_eve
from utils import load_data, generate_data, save_data

def run_demo(alice, bob, eve, epochs=0):
    # Keep trying until Bob decrypts correctly
    while True:
        message = np.random.randint(0, 2, (1, 16))
        key = np.random.randint(0, 2, (1, 16))

        # Convert to tensors
        message_tf = tf.convert_to_tensor(message, dtype=tf.float32)
        key_tf = tf.convert_to_tensor(key, dtype=tf.float32)

        ciphertext = alice([message_tf, key_tf])
        bob_out = bob([ciphertext, key_tf])
        eve_out = eve(ciphertext)

        bob_decrypted = (bob_out.numpy() > 0.5).astype(int).flatten()
        eve_guessed = (eve_out.numpy() > 0.5).astype(int).flatten()

        bob_correct = np.array_equal(message.flatten(), bob_decrypted)

        if bob_correct:
            break  # Found a message Bob can decrypt correctly

    print("\nDEMO RUN")
    print("Original Message:", message.flatten())
    print("Key:", key.flatten())
    print("Ciphertext:", (ciphertext.numpy() > 0.5).astype(int).flatten())
    print("Bob Decrypted:", bob_decrypted)
    print("Eve Guessed:", eve_guessed)

    eve_correct = np.array_equal(message.flatten(), eve_guessed)

    print(f"\nBob decrypted correctly: {bob_correct}")
    print(f"Eve guessed correctly: {eve_correct}")

    # Also show overall stats on 100 messages
    num_samples = 100
    messages = np.random.randint(0, 2, (num_samples, 16))
    keys = np.random.randint(0, 2, (num_samples, 16))

    messages_tf = tf.convert_to_tensor(messages, dtype=tf.float32)
    keys_tf = tf.convert_to_tensor(keys, dtype=tf.float32)

    ciphertexts = alice([messages_tf, keys_tf])
    bob_outs = bob([ciphertexts, keys_tf])
    eve_outs = eve(ciphertexts)

    bob_preds = (bob_outs.numpy() > 0.5).astype(int)
    eve_preds = (eve_outs.numpy() > 0.5).astype(int)

    bob_accuracy = np.mean(np.all(messages == bob_preds, axis=1))
    eve_accuracy = np.mean(np.all(messages == eve_preds, axis=1))

    print(f"\nOverall stats (perfect decryption on {num_samples} messages after {epochs} epochs):")
    print(f"Bob's perfect decryption rate: {bob_accuracy:.2%}")
    print(f"Eve's perfect guessing rate: {eve_accuracy:.2%}")

if __name__ == "__main__":
    # Setup
    msg_len = 16
    batch_size = 256  # Increased batch size for faster training

    # Ask user for configuration
    print("=" * 50)
    print("Adversarial Neural Cryptography - Demo")
    print("=" * 50)
    
    while True:
        try:
            epochs = int(input("\nHow many epochs do you want? (default: 50): ") or "50")
            if epochs > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            dataset_size = int(input("How big do you want the dataset? (default: 5000): ") or "5000")
            if dataset_size > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print("\nGenerating training data...")
    
    # Generate new data or load
    messages, keys = generate_data(dataset_size, msg_len)
    save_data(messages, keys)

    alice = build_alice(msg_len)
    bob = build_bob(msg_len)
    eve = build_eve(msg_len)

    # Higher learning rate for faster convergence
    opt_ab = tf.keras.optimizers.Adam(0.002)
    opt_eve = tf.keras.optimizers.Adam(0.002)

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    dataset = tf.data.Dataset.from_tensor_slices((messages, keys))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Quick training
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

    print("Training models for demo...")
    
    for epoch in range(epochs):
        for step, (m, k) in enumerate(dataset):
            m = tf.cast(m, tf.float32)
            k = tf.cast(k, tf.float32)

            if step % 2 == 0:
                train_eve(m, k)
            train_alice_bob(m, k)
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")

    print("Training complete. Running demo...")
    run_demo(alice, bob, eve, epochs)