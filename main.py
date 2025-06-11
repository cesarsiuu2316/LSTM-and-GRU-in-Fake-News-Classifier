import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # Stemming
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential


# GLOBAL VARIABLES
vocabulary_size = 10000  # Size of the vocabulary


def read_csv_file(path):
    print("Reading CSV file...")
    df = pd.read_csv(path)
    return df


def EDA(df):
    print("Performing EDA...")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())


def preprocess_data(df):
    print("Preprocessing data...")
    df.dropna(inplace=True)  # Remove rows with missing values
    df.drop(columns = ["Unnamed: 0"], inplace=True)  # Drop unnecessary columns
    df.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
    return df


def stem_and_remove_stopwords(content):
    # Download NLTK resources
    nltk.download('stopwords')
    print("\nStemming and removing stopwords...")
    ps = PorterStemmer()
    corpus = []
    for i in range(len(content)):
        review = re.sub('[^a-zA-Z]', ' ', content['title'][i])
        review = review.lower().split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def create_model(max_length):
    print("\nCreating LSTM model...")
    embedding_vector_features = 40  # Size of the embedding vector
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_vector_features, input_shape = (max_length,)))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    return model


def main():
    # Read dataframe
    path = "WELFake_Dataset.csv"
    df = read_csv_file(path)
    df = df.head(1000)

    # Perform EDA and preprocessing
    EDA(df)
    df_clean = preprocess_data(df)

    # Display cleaned data
    print("Cleaned Data:")
    print(df_clean.shape)

    # Get features 
    X = df_clean.drop(columns=["label"])
    y = df_clean["label"]

    # Prepare the corpus by stemming and removing stopwords
    content = X.copy()
    corpus = stem_and_remove_stopwords(content)
    print(f"Corpus: \n{corpus[1]}")

    # One hot encoding representation of the corpus
    onehot_repr = [one_hot(words, vocabulary_size) for words in corpus]
    print(f"One-hot representation: \n{onehot_repr[1]}")

    # Pad sequences to ensure uniform input size
    title_length = 20
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=title_length)
    print(f"Padded sequences: \n{embedded_docs[1]}")

    # Create the LSTM model
    model = create_model(title_length)

    # Convert features and labels to numpy arrays
    X_final = np.array(embedded_docs)
    y_final = np.array(y)
    print(f"X_final shape: {X_final.shape}, y_final shape: {y_final.shape}")

    

if __name__ == "__main__":
    main()