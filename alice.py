from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Concatenate

def build_alice(msg_len=16):
    m = Input(shape=(msg_len,))
    k = Input(shape=(msg_len,))

    x = Concatenate()([m, k])
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(msg_len, activation='sigmoid')(x)

    return Model([m, k], x)