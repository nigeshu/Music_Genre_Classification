
# ============================================================
# MUSIC GENRE CLASSIFICATION USING CNN
# Dataset: GTZAN Music Genre Collection
# Features: Mel Spectrograms
# Explainable AI: Grad-CAM
# Environment: Google Colab
# ============================================================

# -----------------------------
# 1. INSTALL REQUIRED LIBRARIES
# -----------------------------

%pip install -q kagglehub librosa opencv-python

import os
import numpy as np
import pandas as pd
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import files

print("TensorFlow version:", tf.__version__)

# -----------------------------
# 2. DOWNLOAD DATASET
# -----------------------------

path = kagglehub.dataset_download(
    "carlthome/gtzan-genre-collection"
)

print("Dataset downloaded to:", path)

# Locate the genres directory automatically
dataset_path = os.path.join(path, "genres")

if not os.path.isdir(dataset_path):
    for root, dirs, filenames in os.walk(path):
        if all(
            genre in dirs
            for genre in [
                "blues", "classical", "country", "disco",
                "hiphop", "jazz", "metal", "pop",
                "reggae", "rock"
            ]
        ):
            dataset_path = root
            break

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(
        "Could not find the GTZAN genres directory. "
        "Check the downloaded dataset structure."
    )

print("Dataset directory:", dataset_path)
print("Available folders:", os.listdir(dataset_path))

# -----------------------------
# 3. DEFINE MUSIC GENRES
# -----------------------------

genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

# -----------------------------
# 4. DISPLAY SAMPLE MEL SPECTROGRAM
# -----------------------------

file_path = os.path.join(
    dataset_path,
    "blues",
    "blues.00007.au"
)

if os.path.exists(file_path):
    audio, sr = librosa.load(
        file_path,
        duration=30
    )

    print("Sample rate:", sr)
    print("Audio samples:", len(audio))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Blues - Mel Spectrogram")
    plt.tight_layout()
    plt.show()

else:
    print("Sample file not found; continuing with dataset processing.")

# -----------------------------
# 5. PROCESS AUDIO DATA
# -----------------------------

X = []
y = []

for label, genre in enumerate(genres):

    folder = os.path.join(dataset_path, genre)

    if not os.path.isdir(folder):
        print("Missing genre folder:", genre)
        continue

    print(f"Processing {genre}...")

    for file in sorted(os.listdir(folder)):

        if file.lower().endswith(".au"):

            audio_path = os.path.join(folder, file)

            try:
                audio, sr = librosa.load(
                    audio_path,
                    duration=30
                )

                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=128
                )

                mel_db = librosa.power_to_db(
                    mel,
                    ref=np.max
                )

                X.append(mel_db)
                y.append(label)

            except Exception as e:
                print("Error processing", file, e)

print("Finished processing audio.")
print("Total samples:", len(X))

if len(X) == 0:
    raise ValueError(
        "No audio samples were processed. "
        "Check the dataset directory."
    )

# -----------------------------
# 6. RESIZE MEL SPECTROGRAMS
# -----------------------------

X_resized = []

for mel in X:
    mel_resized = cv2.resize(
        mel,
        (128, 128)
    )

    X_resized.append(mel_resized)

X = np.array(X_resized, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 7. NORMALIZE DATA
# -----------------------------

max_abs = np.max(np.abs(X))

if max_abs != 0:
    X = X / max_abs

print("Minimum:", X.min())
print("Maximum:", X.max())

# Add channel dimension for CNN
X = X[..., np.newaxis]

print("Final X shape:", X.shape)

# -----------------------------
# 8. SPLIT DATASET
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 9. BUILD CNN MODEL
# -----------------------------

model = models.Sequential([

    layers.Input(shape=(128, 128, 1)),

    # CNN Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    layers.Dropout(0.3),

    # Ten genre outputs
    layers.Dense(
        10,
        activation="softmax"
    )
])

model.summary()

# -----------------------------
# 10. COMPILE CNN
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 11. TRAIN CNN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 12. EVALUATE MODEL
# -----------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# -----------------------------
# 13. PLOT ACCURACY
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 14. PLOT LOSS
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 15. PREDICT TEST DATA
# -----------------------------

y_pred = model.predict(
    X_test,
    verbose=0
)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

# -----------------------------
# 16. CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred_classes
)

plt.figure(figsize=(10, 8))
plt.imshow(cm)

plt.xticks(
    range(10),
    genres,
    rotation=45
)

plt.yticks(
    range(10),
    genres
)

plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.title("CNN Music Genre Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------
# 17. CLASSIFICATION REPORT
# -----------------------------

report = classification_report(
    y_test,
    y_pred_classes,
    labels=list(range(len(genres))),
    target_names=genres,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_classes,
        labels=list(range(len(genres))),
        target_names=genres,
        zero_division=0
    )
)

genre_report = report_df.loc[genres]

ax = genre_report[
    ["precision", "recall", "f1-score"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("CNN Classification Performance by Genre")
plt.xlabel("Music Genre")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 18. PREDICT AN UPLOADED SONG
# -----------------------------

def prepare_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        duration=30
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_resized = cv2.resize(
        mel_db,
        (128, 128)
    )

    max_abs = np.max(np.abs(mel_resized))

    if max_abs != 0:
        mel_normalized = mel_resized / max_abs
    else:
        mel_normalized = mel_resized

    img_array = mel_normalized[
        np.newaxis,
        ...,
        np.newaxis
    ].astype(np.float32)

    return audio, sr, mel_db, mel_normalized, img_array


def predict_uploaded_song():

    uploaded = files.upload()

    if not uploaded:
        print("No file uploaded.")
        return

    file_path = next(iter(uploaded))

    print("\nProcessing:", file_path)

    audio, sr, mel_db, mel_normalized, img_array = (
        prepare_audio(file_path)
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100

    print("\n" + "=" * 40)
    print("      MUSIC GENRE PREDICTION")
    print("=" * 40)
    print("Song:", file_path)
    print("Genre:", genres[predicted_class].upper())
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 40)

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(
        f"Predicted: {genres[predicted_class].upper()} "
        f"({confidence:.2f}%)"
    )

    plt.tight_layout()
    plt.show()


# -----------------------------
# 19. GRAD-CAM EXPLAINABLE AI
# -----------------------------

# Ensure model is built
_ = model.predict(
    X_test[:1],
    verbose=0
)

print("Model is built successfully.")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)

# Find the last convolutional layer automatically
last_conv_layer_name = next(
    layer.name
    for layer in reversed(model.layers)
    if isinstance(layer, layers.Conv2D)
)

print("Grad-CAM layer:", last_conv_layer_name)


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.layers[-1].output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        heatmap
    )

    return heatmap.numpy()


# -----------------------------
# 20. PREDICTION WITH GRAD-CAM
# -----------------------------

def predict_with_xai(file_path):

    audio, sr, mel_db, mel_normalized, img_array = (
        prepare_audio(file_path)
    )

    print("Audio loaded successfully.")
    print("Sample rate:", sr)
    print("Duration:", len(audio) / sr, "seconds")
    print("CNN input shape:", img_array.shape)

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = np.argmax(prediction)

    confidence = prediction[predicted_class] * 100

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
        predicted_class
    )

    heatmap = cv2.resize(
        heatmap,
        (128, 128)
    )

    print("\n" + "=" * 50)
    print("      MUSIC GENRE CLASSIFICATION")
    print("=" * 50)
    print("Predicted Genre:", genres[predicted_class].upper())
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 50)

    print("\nGenre probabilities:")

    for genre, probability in zip(genres, prediction):
        print(f"{genre:<12} {probability * 100:.2f}%")

    # Original Mel spectrogram
    plt.figure(figsize=(10, 5))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Original Mel Spectrogram")
    plt.tight_layout()
    plt.show()

    # Grad-CAM heatmap overlay
    plt.figure(figsize=(10, 5))

    plt.imshow(
        mel_normalized,
        aspect="auto",
        origin="lower"
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.5,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(label="Importance")

    plt.title(
        "Grad-CAM Explanation | "
        f"Prediction: {genres[predicted_class].upper()}"
    )

    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()
    plt.show()

    return (
        genres[predicted_class],
        confidence,
        prediction,
        heatmap
    )


# -----------------------------
# 21. UPLOAD AUDIO AND RUN XAI
# -----------------------------

uploaded = files.upload()

if uploaded:

    file_path = next(iter(uploaded))

    result = predict_with_xai(file_path)

    predicted_genre, confidence, probabilities, heatmap = result

    print("\nFinal Result")
    print("Genre:", predicted_genre.upper())
    print(f"Confidence: {confidence:.2f}%")

else:
    print("No audio file uploaded.")

# -----------------------------
# END OF PROJECT
# -----------------------------
