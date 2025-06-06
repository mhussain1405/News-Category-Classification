# Dockerfile located in mlops-nlp-project/airflow/dags/
FROM apache/airflow:2.7.2

# Switch to root to install build dependencies
USER root

# Update package lists and install common build tools and Python development headers
# build-essential provides compilers like g++, gcc, make
# python3-dev provides Python.h and other headers needed for compiling C extensions for Python3
# libffi-dev and libssl-dev are common dependencies for various Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to the airflow user to install Python packages
USER airflow

# Upgrade pip first, as the base image might have an older version
RUN pip install --no-cache-dir --user --upgrade pip

# Now install your Python packages
# Using a requirements.txt is often cleaner, but direct install is fine for now.
RUN pip install --no-cache-dir --user \
    "cython<3.0.0" \
    "cymem<2.1.0" \
    "preshed<3.1.0" \
    "murmurhash<1.1.0" \
    pandas \
    scikit-learn \
    nltk==3.8.1 \
    spacy==3.7.2 \
    beautifulsoup4==4.12.3 \
    lxml \
    textblob

# Download NLTK and spaCy models AS THE AIRFLOW USER
RUN python -m nltk.downloader -d /home/airflow/.local/share/nltk_data punkt stopwords wordnet omw-1.4 && \
    python -m spacy download en_core_web_sm

# Optional: Set NLTK_DATA environment variable if models aren't found by default
ENV NLTK_DATA=/home/airflow/.local/share/nltk_data