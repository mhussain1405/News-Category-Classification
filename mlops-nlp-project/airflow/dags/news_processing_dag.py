# airflow/dags/news_processing_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
# from airflow.utils.dates import days_ago # Not strictly needed if using fixed datetime
from airflow.models import Variable

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
from datetime import datetime, timedelta # Ensure datetime and timedelta are imported

# --- Path Setup for utils module ---
# This path is for when the DAG runs inside the Airflow worker container,
# assuming 'src' is mounted to '/opt/airflow/src' as per docker-compose.yaml
PROJECT_SRC_PATH_IN_CONTAINER = '/opt/airflow/src'
if PROJECT_SRC_PATH_IN_CONTAINER not in sys.path:
    sys.path.insert(0, PROJECT_SRC_PATH_IN_CONTAINER)

try:
    from utils.nlp_utils import basic_cleaning, advanced_processing, feature_engineering
except ImportError as e:
    print(f"Error importing nlp_utils: {e}")
    # As a fallback for local testing or if path is an issue, you might try relative,
    # but for Dockerized Airflow, the absolute container path is more reliable.
    # For robustness, ensure nlp_utils is importable or handle this gracefully.
    # This simple print might not be enough if the import fails during DAG parsing by scheduler.
    # A common pattern is to put utility functions directly in the DAG file or a plugins directory
    # if they are small or if managing sys.path becomes too complex.
    # For this project, we assume the mount and sys.path.insert works.
    pass


# --- Configuration (using container paths and Airflow Variables) ---
# These paths are relative to how they are mounted or accessible INSIDE the Airflow worker container.
BASE_DATA_PATH_IN_CONTAINER = '/opt/airflow/data' # As mounted in docker-compose
RAW_DATA_PATH = os.path.join(BASE_DATA_PATH_IN_CONTAINER, 'raw/News_Category_Dataset_v3.json')
PROCESSED_DATA_PATH = os.path.join(BASE_DATA_PATH_IN_CONTAINER, 'processed')
INTERIM_DATA_PATH = os.path.join(BASE_DATA_PATH_IN_CONTAINER, 'interim')

# Using try-except for Airflow Variables to allow DAG to parse if variables are not yet set
# or if running locally outside Airflow for testing (though callables won't run correctly then).
try:
    RANDOM_SEED = int(Variable.get("random_seed", default_var=42))
    TEST_SIZE = float(Variable.get("test_size", default_var=0.2))
    VALIDATION_SIZE = float(Variable.get("validation_size", default_var=0.1))
    SAMPLE_FRACTION = float(Variable.get("sample_fraction", default_var=0.1)) # Added for sampling
except Exception as e:
    print(f"Could not get Airflow Variables for ML params, using defaults: {e}")
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    SAMPLE_FRACTION = 0.1


# Ensure output directories exist (tasks will run in the worker, paths must be valid there)
# This os.makedirs might run during DAG parsing on the scheduler if not careful.
# It's better if tasks ensure their own output directories or if these are pre-existing.
# For now, we'll assume they get created or are okay to be checked at parse time.
# However, tasks should ideally be idempotent and handle directory creation themselves.
# To prevent issues during parsing, moving this into a setup task or the first task is safer.
# For now, let's comment it out here, and tasks can implicitly create them via to_pickle.
# os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
# os.makedirs(INTERIM_DATA_PATH, exist_ok=True)


# --- Python Callable Functions for Tasks ---
def setup_directories_fn(**kwargs):
    print(f"Ensuring directories exist: {PROCESSED_DATA_PATH}, {INTERIM_DATA_PATH}")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(INTERIM_DATA_PATH, exist_ok=True)
    print("Directories ensured.")

def ingest_data_fn(**kwargs):
    ti = kwargs['ti']
    # Ensure directories are there, in case this runs first after a clean slate
    # setup_directories_fn() # Or make setup_directories_fn a separate upstream task

    print(f"Loading data from: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}.")
    
    df = pd.read_json(RAW_DATA_PATH, lines=True)
    df['text'] = df['headline'] + " " + df['short_description']
    df = df[['text', 'category']].dropna()
    
    df_sampled = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)
    print(f"Using a sample of {len(df_sampled)} rows (sample_fraction={SAMPLE_FRACTION}).")

    ingested_data_filepath = os.path.join(INTERIM_DATA_PATH, 'ingested_data.pkl')
    df_sampled.to_pickle(ingested_data_filepath)
    ti.xcom_push(key='ingested_data_filepath', value=ingested_data_filepath)
    print(f"Ingested data saved to {ingested_data_filepath}")

def basic_cleaning_fn(**kwargs):
    ti = kwargs['ti']
    ingested_data_filepath = ti.xcom_pull(task_ids='ingest_data_task', key='ingested_data_filepath')
    if not ingested_data_filepath:
        raise ValueError("Ingested data filepath not found in XComs.")
        
    df = pd.read_pickle(ingested_data_filepath)
    print(f"Performing basic cleaning on {len(df)} rows.")
    df['cleaned_text'] = df['text'].apply(basic_cleaning)
    
    basic_cleaned_data_filepath = os.path.join(INTERIM_DATA_PATH, 'basic_cleaned_data.pkl')
    df[['cleaned_text', 'category']].to_pickle(basic_cleaned_data_filepath)
    ti.xcom_push(key='basic_cleaned_data_filepath', value=basic_cleaned_data_filepath)
    print(f"Basic cleaned data saved to {basic_cleaned_data_filepath}")

def advanced_processing_fn(**kwargs):
    ti = kwargs['ti']
    basic_cleaned_data_filepath = ti.xcom_pull(task_ids='basic_cleaning_task', key='basic_cleaned_data_filepath')
    if not basic_cleaned_data_filepath:
        raise ValueError("Basic cleaned data filepath not found in XComs.")

    df = pd.read_pickle(basic_cleaned_data_filepath)
    print(f"Performing advanced processing on {len(df)} rows.")
    df['processed_text'] = df['cleaned_text'].apply(lambda x: advanced_processing(x, lemmatizer_type='nltk'))
    
    advanced_processed_data_filepath = os.path.join(INTERIM_DATA_PATH, 'advanced_processed_data.pkl')
    df[['processed_text', 'category']].to_pickle(advanced_processed_data_filepath)
    ti.xcom_push(key='advanced_processed_data_filepath', value=advanced_processed_data_filepath)
    print(f"Advanced processed data saved to {advanced_processed_data_filepath}")

def feature_engineering_fn(**kwargs):
    ti = kwargs['ti']
    advanced_processed_data_filepath = ti.xcom_pull(task_ids='advanced_processing_task', key='advanced_processed_data_filepath')
    if not advanced_processed_data_filepath:
        raise ValueError("Advanced processed data filepath not found in XComs.")

    df = pd.read_pickle(advanced_processed_data_filepath)
    print(f"Performing feature engineering on {len(df)} rows.")
    
    engineered_features_df = feature_engineering(df['processed_text'])
    df = pd.concat([df, engineered_features_df], axis=1)
    
    final_features_filepath = os.path.join(INTERIM_DATA_PATH, 'final_features_data.pkl')
    df[['processed_text', 'text_length', 'sentiment_score', 'category']].to_pickle(final_features_filepath)
    ti.xcom_push(key='final_features_filepath', value=final_features_filepath)
    print(f"Feature engineered data saved to {final_features_filepath}")

def split_data_fn(**kwargs):
    ti = kwargs['ti']
    final_features_filepath = ti.xcom_pull(task_ids='feature_engineering_task', key='final_features_filepath')
    if not final_features_filepath:
        raise ValueError("Final features filepath not found in XComs.")

    df = pd.read_pickle(final_features_filepath)
    print(f"Splitting data ({len(df)} rows) into train, validation, and test sets.")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df['category']
    )
    # Adjust validation size calculation relative to the remaining training set size
    effective_validation_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_df, test_size=effective_validation_size, random_state=RANDOM_SEED, stratify=train_df['category']
    )
    
    train_data_path = os.path.join(PROCESSED_DATA_PATH, 'train.pkl')
    val_data_path = os.path.join(PROCESSED_DATA_PATH, 'validation.pkl')
    test_data_path = os.path.join(PROCESSED_DATA_PATH, 'test.pkl')

    train_df.to_pickle(train_data_path)
    val_df.to_pickle(val_data_path)
    test_df.to_pickle(test_data_path)

    ti.xcom_push(key='train_data_path', value=train_data_path)
    ti.xcom_push(key='val_data_path', value=val_data_path)
    ti.xcom_push(key='test_data_path', value=test_data_path)
    
    print(f"Train data ({len(train_df)}) saved to {train_data_path}")
    print(f"Validation data ({len(val_df)}) saved to {val_data_path}")
    print(f"Test data ({len(test_df)}) saved to {test_data_path}")

# --- DAG Definition ---
# It's good practice to set a specific, fixed start_date for scheduled DAGs.
# Choose a date in the past relative to when you want the first *actual* scheduled run to occur.
# E.g., if today is 2024-06-06, and you want the first run for the interval of 2024-06-05
# (which would execute on 2024-06-06 at 02:00 UTC), this start_date is appropriate.
# Adjust year/month/day as needed.
START_DATE_YEAR = 2024
START_DATE_MONTH = 1
START_DATE_DAY = 1

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5), # Slightly increased retry delay
    'start_date': datetime(START_DATE_YEAR, START_DATE_MONTH, START_DATE_DAY),
}

with DAG(
    dag_id='news_category_data_processing_v1',
    default_args=default_args,
    description='DAG for processing news category data: ingestion, cleaning, feature engineering, and splitting.',
    schedule_interval='0 2 * * *',  # Cron expression for daily at 2:00 AM UTC
    catchup=False, # IMPORTANT: Prevents backfilling for past missed schedules
    tags=['nlp', 'data_processing', 'news'],
) as dag:

    start_pipeline = DummyOperator(task_id='start_pipeline')

    # Task to ensure directories exist, especially if INTERIM/PROCESSED might be cleared
    setup_directories_task = PythonOperator(
        task_id='setup_directories_task',
        python_callable=setup_directories_fn,
    )

    ingest_data_task = PythonOperator(
        task_id='ingest_data_task',
        python_callable=ingest_data_fn,
    )

    preprocess_start = DummyOperator(task_id='preprocess_start')
    
    basic_cleaning_task = PythonOperator(
        task_id='basic_cleaning_task',
        python_callable=basic_cleaning_fn,
    )

    advanced_processing_task = PythonOperator(
        task_id='advanced_processing_task',
        python_callable=advanced_processing_fn,
    )

    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=feature_engineering_fn,
    )
    
    preprocess_end = DummyOperator(task_id='preprocess_end')

    split_data_task = PythonOperator(
        task_id='split_data_task',
        python_callable=split_data_fn,
    )

    end_pipeline = DummyOperator(task_id='end_pipeline')

    # --- Define Task Dependencies ---
    start_pipeline >> setup_directories_task >> ingest_data_task >> preprocess_start
    
    preprocess_start >> basic_cleaning_task >> advanced_processing_task >> feature_engineering_task >> preprocess_end
    
    preprocess_end >> split_data_task >> end_pipeline