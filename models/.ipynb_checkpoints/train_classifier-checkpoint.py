import sys
# import libraries
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
from pandas import read_sql_table
import pickle


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    input: SQL filepath for database of saved data(done in ETL step)- filepath is without extension
    Output:
        X: predictor dataframe
        Y: result dataframe
        category_names: list of all possible categories
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath+'.db')
    df=read_sql_table(database_filepath, engine)

    X=df['message']
    y=df.drop(['message','id','original','genre'],axis=1)
    
    category_names = y.columns 
    return X, y, category_names


def tokenize(text):
    """
    Tokenize and parse text field
    
    input: text that needs to be parsed
    Output: tokens extracted from text
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def perf_report(model, X_test, y_test):
    '''
    Create classificartion_report from trained model
    Input: Trained Model and test set
    Output: sklearn's classification report results
    '''
    y_pred = model.predict(X_test)
    
    for jj, column in enumerate(y_test):
        print(column)
        print(classification_report(y_test[column], y_pred[:, jj]))
        

def build_model(X_test, y_test):
    """
    Build model: this trains and fits a model and performs cross validation
    
    input: 
    X_test: predictors
    Y_test: text labels
    Output:
        trained KNN model with n_neighbors chosen with CV
        
    """
    
    pipeline_improved = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('KNN', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    #Train & predict
    pipeline_improved.fit(X_train, y_train)
    
    param_grid =  {'tfidf__use_idf': (True, False), 
              'KNN__estimator__n_neighbors': [2,5,8,11]} 

    cv_improved = GridSearchCV(pipeline_improved, param_grid=param_grid)
    cv_improved.fit(X_train, y_train)

    return(cv_improved)


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate Model: This takes a fit ML pipeline and prints out the model performance report
    
    inputs:
        model: trained ML pipeline
        X_test: predictors
        Y_test: text labels
    """
    perf_report(model, X_test, y_test)


def save_model(model, model_filepath):
    """
    Save model: save trained pipeline model as pickle
    
    inputs:
        model: cross validated model file
        model_filepath: save filepath
    
    """
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()