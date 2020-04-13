import argparse
import sys
sys.path.insert(0, '..')
import json
import pandas as pd
from flask import Flask, request, url_for, redirect, render_template, jsonify
from gensim.models import Word2Vec
from utils import fix_authors, EpochSaver
from coordle_backend import AI_Index

parser = argparse.ArgumentParser(description='Run Coordle Flask app')
parser.add_argument('--port', default=5000, help='port to use when serving the flask app (default: 5000)')
port = parser.parse_args().port

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
num_per_load = 10

def perform_search(query: str, start: int = 0, stop: int = -1):

    # Perform search using AI Index
    docs, _, error_msgs = ai_index.search(query)

    if error_msgs:
        return [error_msgs]
    else:
        # Ensure start/stop is within range
        if start < 0:
            start = 0
        if start > len(docs):
            start = len(docs) - num_per_load
        if stop < num_per_load:
            stop = num_per_load
        if stop > len(docs):
            stop = len(docs)

        # Extract resulting rows from df
        cord_uids = [doc.uid for doc in docs[start:stop]]
        result_df = cord_df[cord_df.cord_uid.isin(cord_uids)]
        result_df.loc[:, 'authors'] = result_df['authors'].apply(fix_authors)
        results = result_df.to_json(orient='records')

        return [results, len(docs)]

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.args.get('q', default='')
    if request.method == 'GET':
        if len(query) > 0:
            return render_template('main.html', searching=True)
        else:
            return redirect('/')
    elif request.method == 'POST':
        start = request.args.get('start', default=0, type=int)
        stop = request.args.get('stop', default=num_per_load, type=int)
        if len(query) > 0:
            
            # Perform search with query limited by start/stop
            search_result = perform_search(query, start, stop)
            if len(search_result) == 2:
                results, total_results, error_msgs = search_result[0], search_result[1], None
            else:
                results, total_results, error_msgs = [], 0, search_result[0]
        else:
            results, total_results, error_msgs = [], 0, None
        return jsonify(results=results, total_results=total_results, error_msgs=error_msgs, num_per_load=num_per_load)

if __name__ == '__main__':

    # Load Word2Vec model and create index for search engine
    w2v_model = Word2Vec.load('web/static/cord-19-w2v.model')
    cord_df = pd.read_csv('web/static/cord-19-data.csv', nrows=100)
    ai_index = AI_Index(w2v_model.wv.most_similar, n_similars=1)
    ai_index.build_from_df(
        df=cord_df,
        uid='cord_uid',
        title='title',
        text='body_text', 
        verbose=True, 
        use_multiprocessing=True,
        workers=-1
    )

    # Run flask app
    app.run(port=port)