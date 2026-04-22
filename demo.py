import numpy as np

def run_demo(alice, bob, eve):
    message = np.random.randint(0, 2, (1, 16))
    key = np.random.randint(0, 2, (1, 16))

    ciphertext = alice([message, key])
    bob_out = bob([ciphertext, key])
    eve_out = eve(ciphertext)

    print("\nDEMO RUN")
    print("Original Message:", message)
    print("Ciphertext:", ciphertext.numpy())
    print("Bob Output:", (bob_out.numpy() > 0.5).astype(int))
    print("Eve Output:", (eve_out.numpy() > 0.5).astype(int))