import json
import pickle
import sys

import numpy as np
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar

sys.path.append(".")
sys.path.append("../")
from models import train_classifier as tc

app = Flask(__name__)

_pipe = None
_clf1 = None
_clf2 = None


def load_model(model_filepath):
    global _pipe, _clf1, _clf2
    with open(model_filepath, 'rb') as f:
        rodelbahn_model = pickle.load(f)
    _pipe = rodelbahn_model["pipe"]
    _clf1 = rodelbahn_model["clf1"]
    _clf2 = rodelbahn_model["clf2"]


def text_prep(text):
    global _pipe
    return _pipe.transform(np.array([text]))


def apply_model(x_transformed):
    global _clf1, _clf2
    labels_zuordnung_mlp = _clf1.classes_
    y_pred_mlp_proba = _clf1.predict_proba(x_transformed)
    y_pred_mlp_label = np.array([labels_zuordnung_mlp[np.argmax(t)] for t in y_pred_mlp_proba])
    y_pred_moc_labels = np.array(_clf2.predict(x_transformed))
    return y_pred_mlp_label, y_pred_moc_labels


# load data
DATA, Y, target = tc.load_data("./data/DisasterResponse.db")
Y = pd.DataFrame(Y)
Y.columns = target

# load model
load_model("./data/rodelbahn_model.pckl")


def show_example():
    t = pd.Series(DATA).sample()
    print(t.iloc[0])
    print(Y.iloc[int(t.index[0])].to_dict())


# index webpage displays 2 visuals and receives user input text for model's prediction
@app.route('/')
@app.route('/index')
def index():
    res = Y.groupby('genre').count()
    genre_counts = list(res.values[:, 0])
    genre_names = list(res.index)

    xdf = Y.copy()
    for xcol in xdf.columns:
        if xcol != 'genre':
            xdf[xcol] = [int(t) for t in xdf[xcol]]

    res = xdf[xdf['genre'] == "direct"].mean().sort_values(ascending=False)
    needs_percentage = res.values.tolist()
    needs_labels = res.index.values.tolist()

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Count of messages per genre',
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
                    x=needs_labels,
                    y=needs_percentage
                )
            ],

            'layout': {
                'title': 'Most current needs for direct messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "",
                    'tickangle': 45
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
    query = request.args.get('query', '')
    y_pred_mlp_label, y_pred_moc_labels = apply_model(text_prep(query))

    # leads to output such as: "Genre: direct", always 1
    classification_results = dict(zip(
        ["Genre: {}".format(y_pred_mlp_label[0])] + target[1:],
        ["1"] + list(y_pred_moc_labels[0])))
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
