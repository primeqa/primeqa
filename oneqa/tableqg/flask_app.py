from tableQG.qg_inference import GenerateQuestions
from tqdm import tqdm
from querycompletion.query_completion import generate
import kenlm

import time
from flask import Flask
from flask import request
from flask import jsonify, make_response, url_for
import json
import sys


app = Flask(__name__)

# Initializing the QG model
# ccc model path
# model_path = '/dccstor/cmv/saneem/nlqTable/irl_git/QG_table/models/t5_col-header_nw-4_if-agg-True_group-g_1/'

# mac model path

try:
    gq = GenerateQuestions(
        './tableQG/models/t5_model/')
except:
    print("cant find the t5 generation model")

try:
    complete_model = kenlm.Model('./resources/wikiSQLdev.arpa')
    print("loaded")
except:
    print("cant find the kenlm completion node")

try:
    with open('./resources/q_dev_questions') as f:
        questions = [q.strip('\n').lower() for q in f]
except:
    print("cant find the generated questions")

try:
    d = 128
    nb = len(questions)

    import numpy as np
    import faiss                   # make faiss available
    index = faiss.IndexFlatL2(d)   # build the index

    for question in tqdm(questions):
        emb = np.random.random((nb, d)).astype(
            'float32')  # gq.model.encode(question)
        index.add(emb)                  # add vectors to the index

    print(index.ntotal)

except:
    print("FAISS not installed correctly")


@app.route('/generate', methods=['POST'])
def predict_tableqa():
    # default params for generation
    num_samples = 5
    # ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
    agg_prob = [1., 0., 0., 0., 0., 0.]
    # [0,1,2,3,4], there is no zero where clause case
    num_where_prob = [0., 1., 0., 0., 0.]
    ineq_prob = 0.0  # probability of inequality coming in where clauses

    if request.is_json:
        req = request.get_json()

        table = req['table']
        if 'num_samples' in req:
            num_samples = int(req['num_samples'])
        if 'agg_prob' in req:
            agg_prob = req['agg_prob']
        if 'num_where_prob' in req:
            num_where_prob = req['num_where_prob']
        if 'ineq_prob' in req:
            ineq_prob = req['ineq_prob']

        question_list = gq.generate_question(
            table, num_samples, agg_prob, num_where_prob, ineq_prob)

        response_body = {'generated_questions': question_list}
        res = make_response(jsonify(response_body), 200)

        return res
    else:
        return make_response(jsonify({"message": "Request body must be JSON"}), 400)


@app.route('/complete', methods=['POST'])
def predict_completion():

    if request.is_json:
        req = request.get_json()
        incomplete = req['query']

        recommend = generate(incomplete, questions, complete_model)

        print(recommend)
        response_body = {'completion_questions': recommend}
        return make_response(jsonify(response_body), 200)

    else:
        return make_response(jsonify({"message": "Request body must be JSON"}), 400)


@app.route('/suggest', methods=['POST'])
def predict_suggestion():

    if request.is_json:
        req = request.get_json()
        query = req['query']
        k = req.get('n', 5)

        try:
            _, I = index.search(query, k)
            NN_questions = questions[I]
            response_body = {'completion_questions': NN_questions}
            return make_response(jsonify(response_body), 200)
        except:
            pass

    return make_response(jsonify({"message": "Request body must be JSON"}), 400)


def has_no_empty_params(rule):
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) >= len(arguments)


@app.route("/")
def site_map():
    links = []
    for rule in app.url_map.iter_rules():
        # Filter out rules we can't navigate to in a browser
        # and rules that require parameters
        if "POST" in rule.methods and has_no_empty_params(rule):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append((url, rule.endpoint))
    # links is now a list of url, endpoint tuples
    return make_response(jsonify(dict(links)), 400)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
