from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(image_shape, learning_rate=1e-3, dropout_rate=0.5):
    """Creates and compiles a CNN for binary classification of cherry-leaf images.
    
    Args:
      image_shape (tuple): e.g. (256, 256, 3)
      learning_rate (float): Adam optimizer step size
      dropout_rate (float): Dropout fraction for the dense layer
    
    Returns:
      tf.keras.Model: compiled CNN ready for training
    """
    model = Sequential([
        Input(shape=image_shape),

        # Block 1
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        # Block 2
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        # Block 3
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        # Classification head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate), # regularization to reduce overfitting
        Dense(1, activation='sigmoid') # single output for binary classification
    ])

    # Compile with documented hyperparameters
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model