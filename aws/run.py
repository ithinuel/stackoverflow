# -*- coding: utf-8 -*-
"""StackOverflow Tag proposition API.

Example:
    The way to use the API is the following::

        $ python run.py

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
from stackoverflow import app

if __name__ == "__main__":

    app.run(host='0.0.0.0', debug=True, port=4000)