import sys
# import libraries
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load data from attached csvs
    input: categories and messages filepath for csv's
    output: loaded and combined data files
    
    """
    # load categories dataset
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    # merge datasets
    df  = categories.merge(messages, how='outer',\
                               on=['id'])

    return df


def clean_data(df):
    """
    clean dataset and create numeric variables for all cateogries
    input: loaded data frame from load_data function
    output: cleaned dataset
    
    """
    # create a dataframe of the 36 individual category columns
    categories=df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    #print(row)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing

    category_colnames = row.str.split('-').str.get(0)
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories = categories.drop('related', axis=1)

    df = df.drop('categories', axis=1)
    df =  pd.concat([df, categories],axis=1)
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    take dataset and save it to database using sql
    input: cleaned dataframe, database filename- filename is without the extension
    output: none, just saves to database
    
    """  
    engine = create_engine('sqlite:///'+database_filename+'.db')
    df.to_sql(database_filename, engine, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()