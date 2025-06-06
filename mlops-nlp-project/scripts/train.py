# scripts/train.py
import pandas as pd
import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'
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
import joblib # For saving sklearn models if not using MLflow's native saving

# --- Configuration ---
# Path to processed data from Airflow (assuming it's relative to project root)
# These paths are on your HOST machine, where this script runs.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/train.pkl')
VALIDATION_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/validation.pkl')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/test.pkl')

MLFLOW_TRACKING_URI = "http://localhost:5000" # Or your remote server if you set one up
EXPERIMENT_NAME = "NewsCategoryClassification_V1"

# --- Helper Functions ---
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Ensure Airflow pipeline has run.")
    return pd.read_pickle(path)

def print_metrics(model_name, metrics):
    print(f"\n--- {model_name} Metrics ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

def get_dataset_version():
    # Placeholder for dataset versioning.
    # Could be a hash of the raw data, a timestamp from Airflow, or a manual version string.
    # For now, let's use the modification timestamp of the training data.
    try:
        return str(time.ctime(os.path.getmtime(TRAIN_DATA_PATH)))
    except FileNotFoundError:
        return "unknown_version"

# --- Main Training Logic ---
if __name__ == "__main__":
    # Set MLflow Tracking URI and Experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or set the experiment
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Experiment '{EXPERIMENT_NAME}' created with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
            print(f"Experiment '{EXPERIMENT_NAME}' already exists with ID: {experiment_id}")
        else:
            raise
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    dataset_version = get_dataset_version()
    print(f"Using Dataset Version: {dataset_version}")

    # 1. Load Data
    print("Loading data...")
    train_df = load_data(TRAIN_DATA_PATH)
    val_df = load_data(VALIDATION_DATA_PATH)
    # test_df = load_data(TEST_DATA_PATH) # Load test_df when ready for final evaluation

    # For simplicity, let's combine train and validation for TF-IDF fitting,
    # then evaluate on the validation set. Proper would be to fit TF-IDF only on train.
    # For now, we'll use val_df for evaluation of these initial models.
    X_train_text = train_df['processed_text']
    y_train = train_df['category']
    
    X_val_text = val_df['processed_text']
    y_val = val_df['category']

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    # Save label encoder for later use (e.g., during inference)
    # It's good practice to log this as an artifact with MLflow too.
    joblib.dump(label_encoder, os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib')) # Create models dir
    print(f"Label encoder saved. Classes: {label_encoder.classes_}")


    # 2. TF-IDF Vectorization (Fit on training data only)
    print("Vectorizing text with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) # Added ngram_range
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_val_tfidf = tfidf_vectorizer.transform(X_val_text)
    # Save the vectorizer
    joblib.dump(tfidf_vectorizer, os.path.join(PROJECT_ROOT, 'models/tfidf_vectorizer.joblib'))
    print("TF-IDF vectorizer saved.")


    # --- Experiment 1: Naive Bayes with TF-IDF ---
    with mlflow.start_run(run_name="NaiveBayes_TFIDF"):
        print("\nTraining Naive Bayes with TF-IDF...")
        mlflow.log_param("model_type", "NaiveBayes")
        mlflow.log_param("features", "TF-IDF")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("tfidf_ngram_range", "(1,2)")
        mlflow.log_param("dataset_version", dataset_version)

        nb_model = MultinomialNB()
        
        start_time = time.time()
        nb_model.fit(X_train_tfidf, y_train_encoded)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        start_time = time.time()
        y_pred_nb = nb_model.predict(X_val_tfidf)
        inference_time = time.time() - start_time
        mlflow.log_metric("inference_time_val_seconds", inference_time)
        
        accuracy_nb = accuracy_score(y_val_encoded, y_pred_nb)
        f1_nb = f1_score(y_val_encoded, y_pred_nb, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy_nb)
        mlflow.log_metric("f1_score_weighted", f1_nb)
        
        print_metrics("Naive Bayes", {"accuracy": accuracy_nb, "f1_score_weighted": f1_nb})

        # Log model
        mlflow.sklearn.log_model(nb_model, "naive_bayes_model")
        # Log TF-IDF vectorizer as an artifact (important for inference pipeline)
        # mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer") # Alternative way
        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/tfidf_vectorizer.joblib'), artifact_path="tfidf_vectorizer_artifact")
        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib'), artifact_path="label_encoder_artifact")

        # Register model (optional, can be done from UI too)
        # mlflow.register_model(
        #     model_uri=f"runs:/{mlflow.active_run().info.run_id}/naive_bayes_model",
        #     name="NewsCategory_NaiveBayes"
        # )
        print("Naive Bayes run complete.")

    # --- Experiment 2: SVM with TF-IDF ---
    with mlflow.start_run(run_name="SVM_TFIDF"):
        print("\nTraining SVM with TF-IDF...")
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("features", "TF-IDF")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("tfidf_ngram_range", "(1,2)")
        mlflow.log_param("dataset_version", dataset_version)
        
        # SVM can be slow on large datasets. Consider a smaller C or LinearSVC for speed.
        # For full dataset, you might need to sample or use more efficient SVM variants.
        # Using Logistic Regression for a faster example here, change to SVC if you prefer and have time.
        # svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        svm_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=200) # Faster alternative
        mlflow.log_param("svm_C", 1.0)
        mlflow.log_param("svm_kernel", "linear" if isinstance(svm_model, SVC) else "N/A for LR")


        start_time = time.time()
        svm_model.fit(X_train_tfidf, y_train_encoded)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        start_time = time.time()
        y_pred_svm = svm_model.predict(X_val_tfidf)
        inference_time = time.time() - start_time
        mlflow.log_metric("inference_time_val_seconds", inference_time)
        
        accuracy_svm = accuracy_score(y_val_encoded, y_pred_svm)
        f1_svm = f1_score(y_val_encoded, y_pred_svm, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy_svm)
        mlflow.log_metric("f1_score_weighted", f1_svm)

        print_metrics("SVM (Logistic Regression)", {"accuracy": accuracy_svm, "f1_score_weighted": f1_svm})
        
        mlflow.sklearn.log_model(svm_model, "svm_model")
        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/tfidf_vectorizer.joblib'), artifact_path="tfidf_vectorizer_artifact")
        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib'), artifact_path="label_encoder_artifact")
        print("SVM (Logistic Regression) run complete.")

    print("\nAll initial experiments complete. Check MLflow UI at http://localhost:5000")

    # TODO: Implement other models as per project requirements:
    # - Word embeddings with simple neural networks (Keras/TensorFlow or PyTorch)
    # - Pre-trained embeddings (Word2Vec, GloVe) with NNs or traditional ML
    # - Simple transformer-based approaches (Hugging Face DistilBERT)




# (Inside train.py, after SVM/LR experiment)
# --- Experiment 3: Simple Neural Network with Own Embeddings ---
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
    mlflow.keras.log_model(nn_model, "simple_nn_keras_model", signature=None) # Add signature later if needed
    mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib'), artifact_path="label_encoder_artifact")
    print("SimpleNN run complete.")




# (Inside train.py, after SimpleNN experiment)
# --- Experiment 4: Neural Network with Pre-trained GloVe Embeddings ---

GLOVE_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'embeddings/glove.6B.100d.txt') # Adjust path
# Ensure GLOVE_EMBEDDING_PATH exists and contains the GloVe file.

if not os.path.exists(GLOVE_EMBEDDING_PATH):
    print(f"GloVe file not found at {GLOVE_EMBEDDING_PATH}. Skipping GloVe experiment.")
else:
    with mlflow.start_run(run_name="NN_GloVe_Embeddings"):
        print("\nTraining Neural Network with Pre-trained GloVe Embeddings...")
        mlflow.log_param("model_type", "NN_GloVe_Embeddings")
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("glove_embedding_file", os.path.basename(GLOVE_EMBEDDING_PATH))

        # Tokenizer and padding (can reuse from previous NN experiment if params are the same,
        # or re-create if MAX_WORDS, MAX_SEQUENCE_LENGTH are different)
        # For simplicity, let's assume we use the same tokenizer settings as SimpleNN
        # tokenizer = joblib.load(os.path.join(PROJECT_ROOT, 'models/keras_tokenizer.joblib')) # Or re-fit if needed
        # X_train_padded, X_val_padded are already available if run in sequence

        MAX_WORDS_GLOVE = MAX_WORDS # Use the same from previous Keras model
        MAX_SEQUENCE_LENGTH_GLOVE = MAX_SEQUENCE_LENGTH
        EMBEDDING_DIM_GLOVE = 100 # Must match the GloVe file (e.g., 100d)

        mlflow.log_param("nn_max_words", MAX_WORDS_GLOVE)
        mlflow.log_param("nn_max_seq_length", MAX_SEQUENCE_LENGTH_GLOVE)
        mlflow.log_param("nn_embedding_dim", EMBEDDING_DIM_GLOVE)

        # Load GloVe embeddings
        embeddings_index = {}
        with open(GLOVE_EMBEDDING_PATH, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print(f"Found {len(embeddings_index)} word vectors in GloVe.")

        # Create embedding matrix
        word_index = tokenizer.word_index # From the previously fitted tokenizer
        embedding_matrix = np.zeros((MAX_WORDS_GLOVE, EMBEDDING_DIM_GLOVE))
        for word, i in word_index.items():
            if i < MAX_WORDS_GLOVE:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        
        # Build Model
        glove_nn_model = Sequential([
            Embedding(input_dim=MAX_WORDS_GLOVE, 
                      output_dim=EMBEDDING_DIM_GLOVE, 
                      weights=[embedding_matrix], 
                      input_length=MAX_SEQUENCE_LENGTH_GLOVE, 
                      trainable=False), # Set trainable=True to fine-tune
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        glove_nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(glove_nn_model.summary())
        mlflow.log_param("nn_glove_trainable_embedding", False)
        mlflow.log_param("nn_optimizer", "adam")
        mlflow.log_param("nn_loss", "sparse_categorical_crossentropy")

        early_stopping_glove = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        mlflow.log_param("nn_early_stopping_patience", 3)

        BATCH_SIZE_GLOVE = 32
        EPOCHS_GLOVE = 10
        mlflow.log_param("nn_batch_size", BATCH_SIZE_GLOVE)
        mlflow.log_param("nn_epochs_config", EPOCHS_GLOVE)


        start_time = time.time()
        history_glove = glove_nn_model.fit(
            X_train_padded, y_train_encoded, # Assumes X_train_padded from previous NN is suitable
            epochs=EPOCHS_GLOVE,
            batch_size=BATCH_SIZE_GLOVE,
            validation_data=(X_val_padded, y_val_encoded), # Assumes X_val_padded from previous NN is suitable
            callbacks=[early_stopping_glove],
            verbose=1
        )
        training_time_glove = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time_glove)
        actual_epochs_glove = len(history_glove.history['loss'])
        mlflow.log_metric("nn_actual_epochs_run", actual_epochs_glove)

        start_time = time.time()
        loss_glove, accuracy_glove_nn = glove_nn_model.evaluate(X_val_padded, y_val_encoded, verbose=0)
        inference_time_val_glove = (time.time() - start_time) / len(X_val_padded)
        mlflow.log_metric("inference_time_val_persample_seconds", inference_time_val_glove)
        
        y_pred_proba_glove_nn = glove_nn_model.predict(X_val_padded)
        y_pred_glove_nn = np.argmax(y_pred_proba_glove_nn, axis=1)
        f1_glove_nn = f1_score(y_val_encoded, y_pred_glove_nn, average='weighted')

        mlflow.log_metric("accuracy", accuracy_glove_nn)
        mlflow.log_metric("f1_score_weighted", f1_glove_nn)
        mlflow.log_metric("val_loss", loss_glove)

        print_metrics("NN_GloVe_Embeddings", {"accuracy": accuracy_glove_nn, "f1_score_weighted": f1_glove_nn, "val_loss": loss_glove})
        
        mlflow.keras.log_model(glove_nn_model, "glove_nn_keras_model", signature=None)
        # Also log the tokenizer used, as it's tied to the word_index for the embedding matrix
        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/keras_tokenizer.joblib'), artifact_path="keras_tokenizer_artifact")
        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib'), artifact_path="label_encoder_artifact")
        print("NN_GloVe run complete.")





# (Inside train.py, after GloVe experiment)
# --- Experiment 5: Simple Transformer (DistilBERT Fine-tuning) ---
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification # TensorFlow version
# Or for PyTorch: from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf # If using TFDistilBertForSequenceClassification
import tf_keras as keras

# For PyTorch version, you'd need to adapt the training loop significantly.
# For simplicity and consistency with Keras above, let's use the TF version from Hugging Face.

# Check if TensorFlow can see a GPU, not strictly necessary for DistilBERT on CPU but good to know
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Using a smaller sample for DistilBERT fine-tuning to speed up demonstration
# In a real scenario, you'd use more data or train for longer.
sample_frac_transformer = 0.1 # Use 10% of the training/val data for this example
train_df_sample_hf = train_df.sample(frac=sample_frac_transformer, random_state=42)
val_df_sample_hf = val_df.sample(frac=sample_frac_transformer, random_state=42)

X_train_hf_texts = train_df_sample_hf['processed_text'].tolist()
y_train_hf_labels = label_encoder.transform(train_df_sample_hf['category']) # Use existing label_encoder

X_val_hf_texts = val_df_sample_hf['processed_text'].tolist()
y_val_hf_labels = label_encoder.transform(val_df_sample_hf['category'])

if not X_train_hf_texts: # check if sampling resulted in empty list
    print("Skipping DistilBERT experiment due to empty training sample after sampling.")
else:
    with mlflow.start_run(run_name="DistilBERT_FineTune"):
        print(f"\nFine-tuning DistilBERT on a sample of {len(X_train_hf_texts)} training texts...")
        mlflow.log_param("model_type", "DistilBERT_FineTune")
        mlflow.log_param("transformer_model_name", "distilbert-base-uncased")
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("training_sample_fraction", sample_frac_transformer)

        MODEL_NAME = 'distilbert-base-uncased'
        tokenizer_hf = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        # Tokenize
        # Max length for DistilBERT is 512, but shorter can be faster if appropriate
        MAX_LENGTH_HF = 128 
        mlflow.log_param("hf_max_length", MAX_LENGTH_HF)

        train_encodings = tokenizer_hf(X_train_hf_texts, truncation=True, padding=True, max_length=MAX_LENGTH_HF, return_tensors="tf")
        val_encodings = tokenizer_hf(X_val_hf_texts, truncation=True, padding=True, max_length=MAX_LENGTH_HF, return_tensors="tf")

        train_dataset_hf = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            y_train_hf_labels
        ))
        val_dataset_hf = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            y_val_hf_labels
        ))

        num_classes_hf = len(label_encoder.classes_)
        
        # Load pre-trained model for sequence classification
        hf_model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes_hf)
        
        # Optimizer and Loss
        # Adam optimizer with weight decay recommended for transformers
        LEARNING_RATE_HF = 5e-5
        # NUM_EPOCHS_HF = 3 # "limited layers" might also imply few epochs
        NUM_EPOCHS_HF = 1 # For quick demo, increase for better results
        BATCH_SIZE_HF = 16 # Adjust based on your memory (8, 16, 32)

        mlflow.log_param("hf_learning_rate", LEARNING_RATE_HF)
        mlflow.log_param("hf_epochs", NUM_EPOCHS_HF)
        mlflow.log_param("hf_batch_size", BATCH_SIZE_HF)

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_HF, epsilon=1e-08)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics_fn = [keras.metrics.SparseCategoricalAccuracy('accuracy')]
        
        hf_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_fn)
        print(hf_model.summary())

        start_time = time.time()
        history_hf = hf_model.fit(
            train_dataset_hf.shuffle(100).batch(BATCH_SIZE_HF),
            epochs=NUM_EPOCHS_HF,
            batch_size=BATCH_SIZE_HF,
            validation_data=val_dataset_hf.batch(BATCH_SIZE_HF)
        )
        training_time_hf = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time_hf)

        start_time = time.time()
        val_loss_hf, val_accuracy_hf = hf_model.evaluate(val_dataset_hf.batch(BATCH_SIZE_HF), verbose=0)
        inference_time_val_hf = (time.time() - start_time) / len(X_val_hf_texts) # Per sample
        mlflow.log_metric("inference_time_val_persample_seconds", inference_time_val_hf)

        # Get F1 score
        y_pred_logits_hf = hf_model.predict(val_dataset_hf.batch(BATCH_SIZE_HF)).logits
        y_pred_hf = np.argmax(y_pred_logits_hf, axis=1)
        f1_hf = f1_score(y_val_hf_labels, y_pred_hf, average='weighted')

        mlflow.log_metric("accuracy", val_accuracy_hf)
        mlflow.log_metric("f1_score_weighted", f1_hf)
        mlflow.log_metric("val_loss", val_loss_hf)

        print_metrics("DistilBERT_FineTune", {"accuracy": val_accuracy_hf, "f1_score_weighted": f1_hf, "val_loss": val_loss_hf})

        # Save and log Hugging Face model with MLflow
        # Need to save tokenizer separately as well for transformers
        # mlflow.transformers.log_model is preferred
        # To use mlflow.transformers.log_model, you usually pass the model and tokenizer objects
        # For TF models, it often involves saving to a directory first then logging that directory.
        
        # Temporary path to save model for logging
        saved_model_path_hf = os.path.join(PROJECT_ROOT, 'models/distilbert_tf_model_temp')
        hf_model.save_pretrained(saved_model_path_hf)
        tokenizer_hf.save_pretrained(saved_model_path_hf) # Save tokenizer in the same directory

        mlflow.log_artifacts(saved_model_path_hf, artifact_path="distilbert_tf_model")
        # Or, using the newer mlflow.transformers module if suitable for your MLflow version
        # try:
        #     import mlflow.transformers
        #     mlflow.transformers.log_model(
        #         transformers_model={'model': hf_model, 'tokenizer': tokenizer_hf},
        #         artifact_path='distilbert_transformers_model',
        #         # input_example=... # Optional: provide an input example for signature
        #     )
        # except ImportError:
        #     print("mlflow.transformers not available, logged as generic artifacts.")
        #     mlflow.log_artifacts(saved_model_path_hf, artifact_path="distilbert_tf_model")


        mlflow.log_artifact(os.path.join(PROJECT_ROOT, 'models/label_encoder.joblib'), artifact_path="label_encoder_artifact")
        print("DistilBERT run complete.")