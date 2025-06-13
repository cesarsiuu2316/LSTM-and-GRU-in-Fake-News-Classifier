import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # Stemming
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import one_hot # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


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


def prepare_data(content):
    # Remove stopwords and stem the content
    corpus = stem_and_remove_stopwords(content)
    print(f"Corpus: \n{corpus[0]}")

    # One hot encoding representation of the corpus
    onehot_repr = [one_hot(words, vocabulary_size) for words in corpus]
    print(f"One-hot representation: \n{onehot_repr[0]}")

    # Pad sequences to ensure uniform input size
    title_length = 20
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=title_length)
    print(f"Padded sequences: \n{embedded_docs[0]}")

    return embedded_docs


def create_model(max_length):
    print("\nCreating LSTM model...")
    embedding_vector_features = 40  # Size of the embedding vector
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_vector_features, input_shape = (max_length,)))
    model.add(LSTM(100))  # Dropout in LSTM
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))) # Regularization to prevent overfitting
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    print("\nTraining the model...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    print("Model training completed.")
    joblib.dump(model, 'models/lstm_model.joblib')


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating the model...")
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to binary predictions
    cm = confusion_matrix(y_test, y_pred)
    accuracy_score_value = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy Score: {accuracy_score_value:.2f}")
    print(f"Classification Report:\n{cr}")


def input_data_for_prediction(model, title):
    docs = prepare_data(pd.DataFrame({'title': [title]}))
    prediction = model.predict(docs)
    prediction = np.where(prediction > 0.5, 1, 0)  # Convert probabilities to binary predictions
    return prediction[0][0]


def main():
    # Read dataframe
    model = None
    try: 
        model = joblib.load('models/lstm_model.joblib')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model not found. A new model will be created.")

    X_test = None
    y_test = None
    if model is None: 
        path = "WELFake_Dataset.csv"
        df = read_csv_file(path)
        df = df.sample(frac=1, random_state=42) # Uncomment to limit the dataset size for testing

        # Perform EDA and preprocessing
        EDA(df)
        df_clean = preprocess_data(df)

        # Display cleaned data
        print("Cleaned Data:")
        print(df_clean.shape)

        # Get features 
        X = df_clean.drop(columns=["label"])
        y = df_clean["label"]

        # Prepare the data
        content = X.copy()
        embedded_docs = prepare_data(content)

        # Create the LSTM model
        title_length = 20
        model = create_model(title_length)

        # Convert features and labels to numpy arrays
        X_final = np.array(embedded_docs)
        y_final = np.array(y)
        print(f"X_final shape: {X_final.shape}, y_final shape: {y_final.shape}")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
        joblib.dump(X_test, 'models/X_test.joblib')
        joblib.dump(y_test, 'models/y_test.joblib')

        # Train the model
        train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate the model
    if X_test is None or y_test is None:
        try: 
            X_test = joblib.load('models/X_test.joblib')
            y_test = joblib.load('models/y_test.joblib')
        except FileNotFoundError:
            print("Test data not found. Please ensure the model is trained first.")
            return
        
    evaluate_model(model, X_test, y_test)

    # Input data for prediction
    salir = False
    while not salir:
        title = input("Enter a title for prediction: ")
        prediction = input_data_for_prediction(model, title)
        print(f"Prediction for '{title}': {'Real' if prediction == 1 else 'False'}")
        salir = input("Do you want to exit? (y/n): ").strip().lower() == 'y'


if __name__ == "__main__":
    main()