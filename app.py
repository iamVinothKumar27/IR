# app.py

from flask import Flask, render_template, request
import ir_models

app = Flask(__name__)

@app.route('/')
def index():
    # Pass all products to the homepage
    return render_template('index.html', products=ir_models.data)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    model = request.form['model']

    if model == 'boolean':
        results = ir_models.boolean_model(query)
        model_name = 'Boolean Model'
    elif model == 'vector':
        results = ir_models.vector_model(query)
        model_name = 'Vector Space Model'
    elif model == 'bm25':
        results = ir_models.bm25_model(query)
        model_name = 'BM25'
    elif model == 'probabilistic':
        results = ir_models.probabilistic_model(query)
        model_name = 'Probabilistic Model'
    else:
        results = []
        model_name = 'Unknown Model'

    return render_template('results.html', results=results, model_name=model_name)

if __name__ == '__main__':
    app.run(debug=True)
