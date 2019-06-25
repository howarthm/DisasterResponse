import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score,classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """load_data
    Load database of messages into a dataframe and
    return the message and labels of categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    return X, Y, Y.columns

def tokenize(text):
    """tokenize
    With the text return it word tokenized
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """build_model
    Use the best_params returned from a GridSearch from an experiment in a jupyter notebook
    Return the model
    """
    clf = RandomForestClassifier(min_samples_split=2,n_estimators=200)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,max_df=0.5,max_features=5000)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(clf)),
    ])
    return pipeline

    
def evaluate_model(model, X_test, Y_test, category_names):
    """evaulate_model
    With the model, test message, and categorized data
    print the precision, recall, and f1-score
    """
    y_pred = model.predict(X_test)

    for i in range(0, len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:, [i]], y_pred[:, i]))



def save_model(model, model_filepath):
    """save_model
    Input the model and the path where to save the model
    and save the model using pickle
    """
    pkl = open(model_filepath, 'wb')
    pickle.dump(model, pkl)
    # Close the pickle file
    pkl.close()


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