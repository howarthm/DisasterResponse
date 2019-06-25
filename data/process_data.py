import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """load_data
    Load the message file, messages_filepath, and categories file, categories_filepath
    and return a merged dataset on column 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    return pd.merge(messages, categories, on="id")

def clean_data(df):
    """clean_data
    Take the dataframe df and return a cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.rsplit('-')[0] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string, convert to a float and coverts any value not a 1 or 0 to a 1
        categories[column] = categories[column].apply(lambda x: 1.0 if float(x.strip()[-1]) > 0 else 0)
    
        # convert column from string to numeric
        #categories[column] = categories[column].apply(lambda x: int(x))

        # some values are not 0 or 1.  Make all the others 1.
        #categories[column] = categories[column].apply(lambda x: 1 if x > 0 else 0)
   

    # drop the original categories column from `df`
    cleandf = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    cleandf = pd.concat([cleandf,categories], axis=1)

    # drop duplicates
    cleandf.drop_duplicates(subset=cleandf.columns.difference(['id']), inplace=True)
    return cleandf
    
def save_data(df, database_filename):
    """save_date
    Save dataframe df to sqlite database database_filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)


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