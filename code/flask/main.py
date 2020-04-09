import json
from flask import Flask, request, render_template
from utils import fix_authors

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')

# Load stuff here

def perform_search(query: str):

    # Mock the results for now
    scores = []
    objects = []
    with open('web/static/results.json', 'r') as file:
        results = json.load(file)

        for obj in results:
            scores.append(42)
            objects.append((obj['cord_uid'], obj['title']))
    
    # TODO: Get rows from cord dataframe with matching cord_uid's

    # TODO: Return results for now
    return results

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('main.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form['query']
    
    # Perform search with query
    results = perform_search(query)
    
    # (scores=[], objs=[(cord_uid, title, word_freq_dict)])

    for result in results:
        result['authors'] = fix_authors(result['authors'])
    
    return render_template('main.html', results=results)

if __name__ == '__main__':
    app.run()