import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Path to data folder
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train, val


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("Loading data...")
    train, val = load_data()

    print("Building model...")
    model = build_model()

    print("Training...")
    history = model.fit(train, validation_data=val, epochs=10)

    model_save_path = os.path.join(os.path.dirname(__file__), "..", "models", "catdog_cnn.h5")
    print("Saving model to:", model_save_path)
    model.save(model_save_path)

    print("Training completed!")


if __name__ == "__main__":
    main()

