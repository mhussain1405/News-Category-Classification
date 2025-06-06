# (Inside train.py, after SVM/LR experiment)
# --- Experiment 3: Simple Neural Network with Own Embeddings ---
import pandas as pd
import os
import mlflow
import mlflow.sklearn # For scikit-learn models
import mlflow.keras # For Keras models
import mlflow.pytorch # For PyTorch models
from sklearn.model_selection import train_test_split # Already split, but useful for local testing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression # Added for another traditional example
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

with mlflow.start_run(run_name="SimpleNN_Embeddings"):
    print("\nTraining Simple Neural Network with Embeddings...")
    mlflow.log_param("model_type", "SimpleNN_Own_Embeddings")
    mlflow.log_param("dataset_version", dataset_version)

    # Tokenization and Padding
    MAX_WORDS = 10000  # Consider words with higher frequency
    MAX_SEQUENCE_LENGTH = 100 # Max length of a sequence
    EMBEDDING_DIM = 100

    mlflow.log_param("nn_max_words", MAX_WORDS)
    mlflow.log_param("nn_max_seq_length", MAX_SEQUENCE_LENGTH)
    mlflow.log_param("nn_embedding_dim", EMBEDDING_DIM)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train_text) # Fit on training text

    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_val_sequences = tokenizer.texts_to_sequences(X_val_text)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Save tokenizer (critical for inference)
    tokenizer_path = os.path.join(PROJECT_ROOT, 'models/keras_tokenizer.joblib')
    joblib.dump(tokenizer, tokenizer_path)
    mlflow.log_artifact(tokenizer_path, artifact_path="keras_tokenizer_artifact")
    print("Keras tokenizer saved and logged.")

    num_classes = len(label_encoder.classes_)

    # Build Model
    nn_model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(nn_model.summary())
    mlflow.log_param("nn_optimizer", "adam")
    mlflow.log_param("nn_loss", "sparse_categorical_crossentropy")

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    mlflow.log_param("nn_early_stopping_patience", 3)
    
    BATCH_SIZE = 32
    EPOCHS = 10 # Or more, with early stopping
    mlflow.log_param("nn_batch_size", BATCH_SIZE)
    mlflow.log_param("nn_epochs_config", EPOCHS)

    start_time = time.time()
    history = nn_model.fit(
        X_train_padded, y_train_encoded,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_padded, y_val_encoded),
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    mlflow.log_metric("training_time_seconds", training_time)
    # Log actual epochs run if early stopping happened
    actual_epochs = len(history.history['loss'])
    mlflow.log_metric("nn_actual_epochs_run", actual_epochs)


    start_time = time.time()
    loss, accuracy_nn = nn_model.evaluate(X_val_padded, y_val_encoded, verbose=0)
    inference_time_val = (time.time() - start_time) / len(X_val_padded) # Per sample
    mlflow.log_metric("inference_time_val_persample_seconds", inference_time_val)

    y_pred_proba_nn = nn_model.predict(X_val_padded)
    y_pred_nn = np.argmax(y_pred_proba_nn, axis=1)
    f1_nn = f1_score(y_val_encoded, y_pred_nn, average='weighted')
    
    mlflow.log_metric("accuracy", accuracy_nn)
    mlflow.log_metric("f1_score_weighted", f1_nn)
    mlflow.log_metric("val_loss", loss)

    print_metrics("SimpleNN_Embeddings", {"accuracy": accuracy_nn, "f1_score_weighted": f1_nn, "val_loss": loss})
    
    # Log Keras model
    mlflow.keras.log_model(nn_model, "simple_nn_keras_model", signature=False) # Add signature later if needed
    mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib'), artifact_path="label_encoder_artifact")
    print("SimpleNN run complete.")