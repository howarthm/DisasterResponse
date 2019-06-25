import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from plotly.graph_objs import Scatter

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    categories = df.drop(['id','message','original','genre'], axis=1)
    category_counts = categories.sum().sort_values()
    
    sorted_category_counts = category_counts.values
    sorted_category_names = category_counts.index

    news = df.loc[df['genre'] == 'news'].drop(['id','message','original','genre'], axis=1)
    news_counts = news.sum()
    social = df.loc[df['genre'] == 'social'].drop(['id','message','original','genre'], axis=1)
    social_counts = social.sum()
    direct = df.loc[df['genre'] == 'direct'].drop(['id','message','original','genre'], axis=1)
    direct_counts = direct.sum()   
    
    # Leave example visualization of genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # Visualization of counts by category
    graphs = [
        {
            'data': [
                Bar(
                    y=sorted_category_names,
                    x=sorted_category_counts,
                    orientation = 'h'
                )
            ],

            'layout': {
                'height': 750,
                'title': 'Counts of categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Counts"
                }
            }
        },
        {
            'data': [
                    Scatter(
                        x=news_counts.index,
                        y=news_counts.values,
                        name='news'
                    ),
                    Scatter(
                        x=direct_counts.index,
                        y=direct_counts.values,
                        name='direct'
                    ),
                    Scatter(
                        x=social_counts.index,
                        y=social_counts.values,
                        name='social'
                    )                
            ],

            'layout': {
                'title': 'Distribution of Counts by Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()