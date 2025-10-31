import pandas as pd

# Load the file as a single column
df = pd.read_csv('./data/sentiment-analysis.csv', header=None)

# Split the first column into multiple columns using commas
df = df[0].str.split(',', expand=True)

# Clean up column names
df.columns = ['Text', 'Sentiment', 'Source', 'Date/Time', 'User ID', 'Location', 'Confidence Score']

# Remove quotes and whitespace
df = df.apply(lambda x: x.str.strip().str.replace('"', ''))

# Save the cleaned dataset
df.to_csv('./data/sentiment-analysis-clean.csv', index=False)
print("Cleaned CSV saved as 'sentiment-analysis-clean.csv'")
