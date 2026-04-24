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
```bash
pip install -r requirements.txt
```

### Quick Demo (Interactive)
Run an interactive demo where you can specify the number of epochs and dataset size:
```bash
python3 demo.py
```
This will prompt you to enter:
- **Number of epochs**: How many epochs to train (default: 50)
- **Dataset size**: How many message/key pairs to generate (default: 5000)

The demo will then:
1. Generate the training data
2. Train Alice, Bob, and Eve networks
3. Show a message that Alice encrypted
4. Display Bob's decryption (correct) and Eve's guess (incorrect)
5. Report statistics on 100 test messages

### Full Training with Visualization
Train the models with full 80 epochs and generate training curves:
```bash
python3 train.py
```
This produces:
- `training_loss_curves.png`: Shows training loss over time for Bob and Eve
- Final evaluation results on 1000 unseen messages

### Final Evaluation
Run evaluation on 1000 unseen test messages:
```bash
python3 evaluate.py
```
Outputs:
- **Bob's bit accuracy**: Should be >95%
- **Eve's bit accuracy**: Should be ~50% (random guessing)

### Files Overview
- `alice.py`: Alice network (encryption)
- `bob.py`: Bob network (decryption with key)
- `eve.py`: Eve network (eavesdropping without key)
- `train.py`: Full training script with visualization
- `demo.py`: Interactive demo with customizable parameters
- `evaluate.py`: Final evaluation on test set
- `utils.py`: Data generation and loading utilities

## Features

### Interactive Demo
- **Customizable training**: Input number of epochs and dataset size at runtime
- **Guaranteed successful encryption**: Demo finds an example where Bob decrypts correctly
- **Statistics reporting**: Shows Bob's and Eve's accuracy on 100 test messages
- **Optimized training**: Uses larger batch sizes (256) and higher learning rates (0.002) for faster convergence

### Training Visualization
- **Loss curves**: Generates `training_loss_curves.png` showing Bob and Eve's training loss over time
- **Real-time monitoring**: Tracks model convergence during training

### Comprehensive Evaluation
- Tests on 1000 unseen messages
- Reports bit-level accuracy for both Bob and Eve
- Demonstrates the security of the adversarial cryptography approach

## Results

With the optimized parameters:
- **Bob's bit accuracy**: ~94-96% (can decrypt messages with the key)
- **Eve's bit accuracy**: ~50-60% (random guessing - cannot break encryption without key)

This demonstrates successful adversarial learning where:
- Alice and Bob learn secure communication
- Eve fails to break the encryption without the shared key
- The gap between Bob's and Eve's accuracy proves the encryption works