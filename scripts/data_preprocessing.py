import json
import pandas as pd

# Load the dataset
with open('dataset/dataset.json', 'r') as f:
    data = json.load(f)

# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# merge the location and description to create a training text.
df['text'] = df['location'] + ": " + df['description']

# Save the preprocessed text as a CSV (or text file)
df['text'].to_csv('dataset/preprocessed_data.csv', index=False, header=False)

print("Data preprocessing complete. Preprocessed data saved to 'dataset/preprocessed_data.csv'.")
