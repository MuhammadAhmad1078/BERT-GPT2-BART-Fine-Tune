import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./data/sentiment-analysis.csv')
print(df.columns.tolist())

# Keep only necessary columns
df = df[['Text', 'Sentiment']]

df['Text'] = df['Text'].str.strip()
df['Sentiment'] = df['Sentiment'].str.strip()

# Encode labels (Positive=2, Neutral=1, Negative=0)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentiment'])

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
train_df.head()
# Save the preprocessed splits
train_df.to_csv('./data/train_preprocessed.csv', index=False)
test_df.to_csv('./data/test_preprocessed.csv', index=False)

print("Saved preprocessed files:")
print("   → ./data/train_preprocessed.csv")
print("   → ./data/test_preprocessed.csv")