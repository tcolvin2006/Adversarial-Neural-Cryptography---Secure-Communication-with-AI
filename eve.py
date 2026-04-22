from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

def build_eve(msg_len=16):
    c = Input(shape=(msg_len,))

    x = Dense(8, activation='relu')(c)
    x = Dense(msg_len, activation='sigmoid')(x)

    return Model(c, x)