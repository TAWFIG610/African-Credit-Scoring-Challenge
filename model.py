import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

def create_model(input_dim):
    model = Sequential([
        Dense(512, input_dim=input_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
