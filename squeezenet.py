"""
Script d'entraînement pour reproduire SqueezeNet sur CIFAR-10, avec :
- Chargement et prétraitement du dataset CIFAR-10.
- Définition du modèle SqueezeNet avec Fire Modules.
- Callbacks : EarlyStopping, ReduceLROnPlateau et TensorBoard.
- Entraînement et évaluation du modèle.
"""

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, concatenate,
                                     GlobalAveragePooling2D, Dropout, Activation)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical

# Pour la reproductibilité
np.random.seed(14)
tf.random.set_seed(14)

# --- Fire Module ---
def fire_module(x, squeeze_filters, expand_filters):
    """
    Module Fire :
    - Couche squeeze : Convolution 1x1 pour réduire la dimensionnalité.
    - Couches expand : Convolutions 1x1 et 3x3 pour enrichir les représentations,
      leurs sorties sont concaténées.
    """
    squeeze = Conv2D(squeeze_filters, (1, 1), padding='valid', activation='relu')(x)
    expand_1x1 = Conv2D(expand_filters, (1, 1), padding='valid', activation='relu')(squeeze)
    expand_3x3 = Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(squeeze)
    x = concatenate([expand_1x1, expand_3x3], axis=-1)
    return x

# --- Construction du modèle SqueezeNet ---
def build_squeezenet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    # Couche initiale
    x = Conv2D(96, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Bloc Fire Modules
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Dernier bloc Fire Modules
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)

    # Classification finale
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, (1, 1), padding='valid', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs, name='SqueezeNet')
    return model

# --- Chargement et prétraitement du dataset CIFAR10 ---
def load_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalisation des images
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # Conversion des labels en one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test, num_classes=10)
    
    # Séparation de la validation (10 % des données d'entraînement)
    val_fraction = 0.1
    num_val = int(x_train.shape[0] * val_fraction)
    x_val = x_train[:num_val]
    y_val = y_train[:num_val]
    x_train = x_train[num_val:]
    y_train = y_train[num_val:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# --- Configuration et lancement de l'entraînement ---
def main():
    # Préparation des données
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_preprocess_data()
    
    # Construction du modèle
    model = build_squeezenet(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()

    # Compilation du modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
    callbacks = [tensorboard_callback, early_stopping, lr_scheduler]
    
    # Entraînement complet
    history = model.fit(x_train, y_train,
                        epochs=100,
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)
    
    # Évaluation finale sur le jeu de test
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f} / Test accuracy: {test_accuracy:.4f}")
    
    # Sauvegarde du modèle
    model.save("squeezenet_cifar10.h5")

if __name__ == "__main__":
    main()
