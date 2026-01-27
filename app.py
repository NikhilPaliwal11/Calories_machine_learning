from flask import Flask, render_template, request
import pandas as pd
import os

from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
# Flask(__name__) it will gives us the entry point where we need to execute it
# Flask(__name__) tells Flask where the application is located so it can find templates, static files, and resources correctly.


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
# User visits page in browser
# Loads templates/index.html

@app.route("/predict", methods=["POST"])
# POST .. Where user will select different values and using this we get the input from user and try to predict the output.
def predict():
    data = CustomData(
        Gender=request.form["Gender"],
        Age=float(request.form["Age"]),
        Height=float(request.form["Height"]),
        Weight=float(request.form["Weight"]),
        Duration=float(request.form["Duration"]),
        Heart_Rate=float(request.form["Heart_Rate"]),
        Body_Temp=float(request.form["Body_Temp"])
    )

    pred_df = data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    result = int(round(pipeline.predict(pred_df)[0]))

    return render_template(
        "result.html",
        calories=result
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

'''
Render template: It connects your backend logic with your frontend page.
Load result.html and replace {{ calories }} with the value of prediction.
'''