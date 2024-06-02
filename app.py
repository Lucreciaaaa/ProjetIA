# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


from flask import Flask, request, jsonify
import numpy as np


app = Flask(__name__)

# Exemple de modèle de prédiction
def predict_market(data):
    # Modèle de prédiction fictif
    return {"prediction": np.random.random()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = predict_market(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)