import json
from flask import Flask, request, url_for, redirect, render_template, jsonify
from utils import fix_authors

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
num_per_load = 5 # TODO: Change this before production

def perform_search(query: str, start: int = 0, stop: int = -1):

    # TODO: Get rows from cord dataframe with matching cord_uid's
    # From coordle api: (scores=[], objs=[(cord_uid, title, word_freq_dict)])

    # Mock the results for now
    scores = []
    objects = []
    with open('web/static/results.json', 'r') as file:
        results = json.load(file)
        for result in results:
            result['authors'] = fix_authors(result['authors'])

        #for obj in results:
        #    scores.append(42)
        #    objects.append((obj['cord_uid'], obj['title']))

    # Ensure start/stop is within range
    if start < 0:
        start = 0
    if start > len(results) - num_per_load:
        start = len(results) - num_per_load
    if stop < num_per_load:
        stop = num_per_load
    if stop > len(results):
        stop = len(results)

    return results[start:stop], len(results)

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
            results, total_results = perform_search(query, start, stop)
        else:
            results, total_results = [], 0
        return jsonify(results=results, total_results=total_results, num_per_load=num_per_load)

if __name__ == '__main__':
    app.run()