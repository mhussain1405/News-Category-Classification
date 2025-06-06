import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# The dataset is a JSON file where each line is a valid JSON object.
file_path = '../data/raw/News_Category_Dataset_v3.json'
df = pd.read_json(file_path, lines=True)

print("Dataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCategory distribution:")
category_counts = df['category'].value_counts()
print(category_counts)

# Plot category distribution
plt.figure(figsize=(12, 8))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=90)
plt.title("Distribution of News Categories")
plt.xlabel("Category")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()

# Combine headline and short_description for text analysis
df['text'] = df['headline'] + " " + df['short_description']
print("\nSample combined text:")
print(df[['text', 'category']].head())