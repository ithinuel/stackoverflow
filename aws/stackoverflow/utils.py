# -*- coding: utf-8 -*-
"""

This module is in charge of model management use for StackOverflow
tag proposition. It provides API to propose tags for questions.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
from __future__ import print_function
import sys
import datetime
import os
import pandas as pd
import numpy as np


class StackOverflow:
    """Model class definition.
    """

    def __init__(self, model, vectorizer):
        tp = pd.read_csv(
            os.path.join('stackoverflow', 'static', 'dataset_clean2.csv'),
            sep='\t', chunksize=1000)
        self.df_test = pd.concat(tp, ignore_index=True)
        """df_test: Loaded test dataset.
        """
        self.model = model
        """model: Loaded model.
        """
        self.vectorizer = vectorizer
        """vectorizer: Loaded vectorizer.
        """

    def getQuestions(self):
        """Return all questions title from the test dataset
        """
        return self.df_test['Title']

    def predict(self, question):
        """Return data matrix with three topics predicted
        """
        subset = self.df_test[self.df_test.Title == question]
        # Predict topic value
        X = subset.raw_letter.values
        tf = self.vectorizer.transform(X)
        X_topic = self.model.transform(tf)
        topic = X_topic[0].argsort()[-3:][::-1]
        # Once topic is extracted, get attributed tags
        tag_list = self.__get_tags_from_topic(topic[0])
        # Build data matrix to return to the upper layer
        data = {}
        data['title'] = subset.Title.values[0]
        data['body'] = subset.raw.values[0][:200]
        data['tags'] = subset.y.values[0]
        data['tag_0'] = tag_list[0]
        data['tag_1'] = tag_list[1]
        data['tag_2'] = tag_list[2]
        return data

    # _________________________ PRIVATE FUNCTIONS _____________________________
    def __get_tags_from_topic(self, topic):
        """Return 3 top tags from topic index.
        """
        if (topic < 0) or (topic > 19):
            return ['None', 'None', 'None']
        tag_mapping = [
            ['android', 'plugin', 'php'],
            ['html', 'script', 'js'],
            ['np', 'array', 'python'],
            ['android', 'google', 'https'],
            ['array', 'np', 'json'],
            ['py', 'python', 'request'],
            ['js', 'angular', 'node'],
            ['server', 'request', 'http'],
            ['view', 'array', 'python'],
            ['android', 'layout', 'java'],
            ['script', 'html', 'js'],
            ['java', 'http', 'android'],
            ['array', 'node', 'google'],
            ['tf', 'np', 'python'],
            ['np', 'plt', 'array'],
            ['view', 'np', 'array'],
            ['request', 'json', 'http'],
            ['array', 'server', 'request'],
            ['array', 'react', 'request'],
            ['php', 'database', 'array']
        ]
        return tag_mapping[topic]
