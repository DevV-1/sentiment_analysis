import pandas as pd

# Loading the dataset
data = pd.read_csv('sentiment_tweets3.csv')

# Inspecting the dataset
print(data.head())

# Preprocessing the data
data.dropna(inplace=True)
data = data.rename(columns = {'label (depression result)' : 'label'})

# Features and target variable
X = data['message to examine']
y = data['label']

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)


from sklearn.metrics import classification_report

# Predict the test set results
y_pred = pipeline.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))


import pickle

# Save the model
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
