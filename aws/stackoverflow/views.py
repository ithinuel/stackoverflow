# -*- coding: utf-8 -*-
"""

This module is in charge of defining routes for Flask Application.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
from __future__ import print_function
import sys
import os
from flask import Flask, render_template, url_for, request, jsonify
import json
import numpy as np
from sklearn.externals import joblib
import dill as pickle
import __main__
from .utils import StackOverflow

app = Flask(__name__)
app.config.from_object('config')
"""app: Flask Application Object.
"""
model_object = pickle.load(
        open(os.path.join('stackoverflow', 'static', 'model.pkl'), 'rb'))
"""model_object: Load Model from static folder.
"""
vectorizer_object = pickle.load(
        open(os.path.join('stackoverflow', 'static', 'vectorizer.pkl'), 'rb'))
"""vectorizer_object: Load Model from static folder.
"""
mapping_object = pickle.load(
        open(os.path.join('stackoverflow', 'static', 'mapping.pkl'), 'rb'))
"""vectorizer_object: Load Model from static folder.
"""
myStackOverflow_object = StackOverflow(model_object, vectorizer_object, mapping_object)
"""myStackOverflow_object: StackOverflow Class Object.
"""


@app.route('/')
@app.route('/index/')
def index():
        """Flask Route for Index.
        """
        return render_template(
                'index.html',
                list_questions=myStackOverflow_object.getQuestions())


@app.route('/result/', methods=['GET', 'POST'])
def result():
        """Flask Route for result.
        """
        question = request.form.get('question_select')
        return render_template(
                  'result.html',
                  result=myStackOverflow_object.predict(question))
