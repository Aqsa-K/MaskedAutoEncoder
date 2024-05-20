import os
import json
import io
import numpy as np
from google.cloud import storage

# This creates a client that uses the specified service account credentials
client = storage.Client()
# Now you can interact with Google Cloud Storage using this client

# Lists all the buckets
buckets = list(client.list_buckets())
print(buckets)


from google.cloud import storage
import pickle

def load_data_from_gcs(bucket_name, x_test_blob_name):
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blobs (files) from the bucket
    x_test_blob = bucket.blob(x_test_blob_name)
    # y_test_blob = bucket.blob(y_test_blob_name)

    # Download the blobs as strings
    x_test_data = x_test_blob.download_as_string()
    # y_test_data = y_test_blob.download_as_string()

    # Load the data using pickle
    X_test = pickle.loads(x_test_data)
    # y_test_encoded = pickle.loads(y_test_data)

    return X_test

# Usage
bucket_name = 'timit_data'
x_test_blob_name = 'processed_data/data_X.pkl'
y_test_blob_name = 'processed_data/data_y.pkl'

X = load_data_from_gcs(bucket_name, x_test_blob_name)
y_encoded = load_data_from_gcs(bucket_name, y_test_blob_name)


phoneme_to_int = load_data_from_gcs(bucket_name, 'processed_data/my_dict.pkl')


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create generator function
def generator(data_X, data_y):
    for features, labels in zip(data_X, data_y):
        yield features, labels

# Create training and validation datasets
train_dataset = tf.data.Dataset.from_generator(lambda: generator(X_train, y_train), output_signature=(
    tf.TensorSpec(shape=(None, 13), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)
))

val_dataset = tf.data.Dataset.from_generator(lambda: generator(X_val, y_val), output_signature=(
    tf.TensorSpec(shape=(None, 13), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)
))

# Apply padding and batching to both datasets
padded_shapes = ([None, 13], [None])
padding_values = (tf.constant(0, dtype=tf.float32), tf.constant(phoneme_to_int['pad'], dtype=tf.int32))

train_dataset = train_dataset.padded_batch(32, padded_shapes=padded_shapes, padding_values=padding_values)
val_dataset = val_dataset.padded_batch(32, padded_shapes=padded_shapes, padding_values=padding_values)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.0, input_shape=(None, 13)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(phoneme_to_int), activation='softmax'))
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=70, validation_data=val_dataset, callbacks=[early_stopping, lr_scheduler])
#model.fit(train_dataset, epochs=100, validation_data=val_dataset)
