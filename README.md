# Adversarial-Neural-Cryptography---Secure-Communication-with-AI
## Overview
This project uses three neural networks:
- Alice encrypts messages using a key
- Bob decrypts the messages
- Eve tries to break the encryption

## How it works
Alice and Bob learn to communicate securely while Eve tries to predict the original message without the key.

## Results

Bob achieved a high accuracy of about 98%, showing that he successfully learned to decrypt the messages. Eve achieved around 63% accuracy, which is significantly lower than Bob but still above random guessing.

## Explanation

This shows that the adversarial training was partially successful. Bob clearly learned the communication task, while Eve was not able to fully recover the original message without the key.

## Limitation

Eve’s accuracy being above 50% suggests that the encryption is not perfectly secure. This is a limitation of the simple model and training setup used in this project.

## How to run

Install dependencies:
pip install -r requirements.txt

Generate data:
python generate_data.py

Train model:
python train.py

## report
This project used three neural networks: Alice, Bob, and Eve. Alice learns to encrypt messages using a key, Bob learns to decrypt them, and Eve tries to guess the message without the key.

During training, Bob’s accuracy increased while Eve stayed close to random guessing. This shows that Alice and Bob successfully learned to communicate while limiting Eve’s ability to decode the message.

However, this is only a simulation and not real encryption. It does not provide real-world security.

Overall, the project demonstrates basic adversarial learning between neural networks.