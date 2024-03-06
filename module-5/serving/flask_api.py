from flask import Flask, jsonify, request
from serving.predictor import Predictor

app = Flask(__name__)
predictor = Predictor.default_from_model_registry()


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json['text']
    result = predictor.predict(payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
