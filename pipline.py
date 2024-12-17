import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -------------------------------
# PHASE 1: DATA PREPROCESSING
# -------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.5, highcut=10.0, fs=50):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def generate_spectrogram(signal, fs=50, nperseg=128):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    return f, t, Sxx

def process_data(data_path, output_dir, fs=50, window_size=2.56, overlap=0.5):
    data = pd.read_parquet(data_path)
    os.makedirs(output_dir, exist_ok=True)

    window_samples = int(window_size * fs)
    step_size = int(window_samples * (1 - overlap))

    results = []

    for start in range(0, len(data) - window_samples, step_size):
        end = start + window_samples
        window = data.iloc[start:end]

        acc_x = apply_bandpass_filter(window['Acc_X'].values)
        acc_y = apply_bandpass_filter(window['Acc_Y'].values)
        acc_z = apply_bandpass_filter(window['Acc_Z'].values)

        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        f, t, Sxx = generate_spectrogram(magnitude, fs=fs)

        plt.figure(figsize=(5, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.axis('off')

        label = window['label'].mode()[0]  # Use the majority label in the window
        file_name = f"spectrogram_{start}_{label}.png"
        plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight', pad_inches=0)
        plt.close()

        results.append((file_name, label))

    return pd.DataFrame(results, columns=['file_name', 'label'])

# -------------------------------
# PHASE 2: DATASET PREPARATION
# -------------------------------

def prepare_dataset(spectrogram_dir, csv_path):
    data = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in data.iterrows():
        img_path = os.path.join(spectrogram_dir, row['file_name'])
        label = row['label']

        if os.path.exists(img_path):
            img = plt.imread(img_path)
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, 128, 128, 1)  # Assuming 128x128 grayscale images
    labels = to_categorical(labels, num_classes=2)  # Binary classification

    return train_test_split(images, labels, test_size=0.2, random_state=42)

# -------------------------------
# PHASE 3: MULTITASK CNN
# -------------------------------

def build_multitask_model(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    tremor_output = Dense(1, activation='sigmoid', name='tremor_output')(x)
    context_output = Dense(3, activation='softmax', name='context_output')(x)

    model = Model(inputs=input_layer, outputs=[tremor_output, context_output])
    model.compile(
        optimizer=Adam(learning_rate=0.0046),
        loss={'tremor_output': 'binary_crossentropy', 'context_output': 'categorical_crossentropy'},
        metrics={'tremor_output': 'accuracy', 'context_output': 'accuracy'}
    )

    return model

# -------------------------------
# PHASE 4: TRAINING
# -------------------------------

def train_model(model, x_train, y_train_tremor, y_train_context, x_val, y_val_tremor, y_val_context):
    history = model.fit(
        x_train, {'tremor_output': y_train_tremor, 'context_output': y_train_context},
        validation_data=(x_val, {'tremor_output': y_val_tremor, 'context_output': y_val_context}),
        epochs=200,
        batch_size=64
    )
    return history

# -------------------------------
# RUN PIPELINE
# -------------------------------

data_path = "processed_all_movement_data.parquet"
spectrogram_dir = "spectrograms"
output_csv = "spectrogram_labels.csv"

# Phase 1: Preprocessing
data_labels = process_data(data_path, spectrogram_dir)
data_labels.to_csv(output_csv, index=False)

# Phase 2: Dataset Preparation
x_train, x_val, y_train, y_val = prepare_dataset(spectrogram_dir, output_csv)

# Split multitask labels
y_train_tremor = y_train[:, 0]
y_train_context = y_train[:, 1:]
y_val_tremor = y_val[:, 0]
y_val_context = y_val[:, 1:]

# Phase 3: Build Model
model = build_multitask_model(input_shape=(128, 128, 1))

# Phase 4: Train Model
history = train_model(model, x_train, y_train_tremor, y_train_context, x_val, y_val_tremor, y_val_context)

print("Pipeline completata con successo!")
