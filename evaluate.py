import numpy as np

def bit_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred)

def evaluate(alice, bob, eve):
    messages = np.random.randint(0, 2, (1000, 16))
    keys = np.random.randint(0, 2, (1000, 16))

    c = alice([messages, keys])
    bob_pred = bob([c, keys])
    eve_pred = eve(c)

    print("\nFINAL EVALUATION")
    print("Bob Accuracy:", bit_accuracy(messages, bob_pred.numpy()))
    print("Eve Accuracy:", bit_accuracy(messages, eve_pred.numpy()))