# **LSTM and GRU in Fake News Classifier**

## **Prerequisites**
**Python version 3.8 - 3.11** must be installed on your system.

## **Steps to Run the Project**

### 1. Clone the repository and download the dataset

```bash
git clone https://github.com/cesarsiuu2316/LSTM-and-GRU-in-Fake-News-Classifier.git
```
Download the csv file [Fake_News_Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place it into the root folder

### 2. Create and Activate a Virtual Environment
It is recommended to use a virtual environment to install dependencies. Follow these steps:

```bash
# Create the virtual environment using pip
python -m venv venv
```
or

```bash
# if you have multiple python versions
python3.11 -m venv venv
```

Activate the virtual environment
```bash
# Activate environment
venv\Scripts\activate
```

### 3. **Install Dependencies**
Once the virtual environment is active, use the `requirements.txt` file from the repository to install the necessary dependencies. Run the following command:

```bash
pip install -r requirements.txt
```

### 4. Finally, run the following python programs:

```bash
python main.py
```