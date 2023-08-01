import sys

import pandas as pd

sys.path.append("../")
sys.path.append(".")
from heimat.eingang.dq_csv import CSV

from sqlalchemy import create_engine

messages = None
categories = None

df_total = None


def load_data(msg_fpath, categ_fpath):
    global messages, categories
    global df_total
    messages = CSV(msg_fpath, encoding="utf-8", sep=",", skiprows=0)
    messages.lesen()
    messages.data.head()
    categories = CSV(categ_fpath, encoding="utf-8", sep=",", skiprows=0)
    categories.lesen()
    categories.data.head()
    xs = pd.Series(categories.data.categories).str.split(pat=";", expand=True)
    xcolumns = [t.split('-')[0] for t in xs.iloc[0].values]
    xfwert = lambda xrecord: [int(t.split("-")[1]) for t in xrecord.values]
    xdf = xs.apply(xfwert, axis=0)
    xdf.columns = xcolumns
    df_total = pd.concat([messages.data, xdf], axis=1)


def clean_data():
    global df_total
    df_total = df_total.drop_duplicates()


def save_data(db_fpath):
    global df_total
    engine = create_engine('sqlite:///{}'.format(db_fpath))
    df_total.to_sql('msg_tbl', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        clean_data()

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


"""
python ./data/process_data.py ./data/disaster_messages.csv  ./data/disaster_categories.csv data/DisasterResponse.db
"""
if __name__ == '__main__':
    main()

