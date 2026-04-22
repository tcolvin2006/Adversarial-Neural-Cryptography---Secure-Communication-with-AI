import numpy as np

def generate_data(num_samples=5000, msg_len=16):
    messages = np.random.randint(0, 2, (num_samples, msg_len))
    keys = np.random.randint(0, 2, (num_samples, msg_len))
    return messages, keys

def save_data(messages, keys):
    np.save("messages.npy", messages)
    np.save("keys.npy", keys)

def load_data():
    messages = np.load("messages.npy")
    keys = np.load("keys.npy")
    return messages, keys