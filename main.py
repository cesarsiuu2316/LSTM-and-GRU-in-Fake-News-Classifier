import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential


# GLOBAL VARIABLES
vocabulary_size = 10,000  # Size of the vocabulary


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


def main():
    # Read dataframe
    path = "WELFake_Dataset.csv"
    df = read_csv_file(path)

    # Perform EDA and preprocessing
    EDA(df)
    df_clean = preprocess_data(df)

    # Display cleaned data
    print("Cleaned Data:")
    print(df_clean.shape)

    # Get features 
    X = df_clean.drop(columns=["label"])
    y = df_clean["label"]

    print(tf.__version__)



if __name__ == "__main__":
    main()