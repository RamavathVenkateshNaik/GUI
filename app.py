
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
from models import db, PredictionResult
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from model import predict_parameter, physical_model

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

a = 5.43e-10
tsi = 2.17

with app.app_context():
    db.create_all()

def generate_plot(x, y_column, y_pred):
    fig, ax = plt.subplots()
    ax.plot(x, y_column, label="Original Y", color="black", marker='o', markersize=3)
    ax.plot(x, y_pred, label="Predicted Y", color="red", linestyle='--', marker='s', markersize=3)
    ax.set_title("Predicted vs Original Y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    img = io.BytesIO()
    plt.tight_layout()
    fig.savefig(img, format="png")
    img.seek(0)
    plot_url = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    thickness = None

    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            voltage = float(request.form["voltage"])
            layers = int(request.form["layers"])
            n_max = float(request.form["nmax"])
            thickness = (layers - 1) * (a / 4)
            result = PredictionResult.query.filter_by(
                temperature=temperature,
                voltage=voltage,
                layers=layers,
                n_max=n_max,
                thickness=thickness
            ).order_by(PredictionResult.id.desc()).first()

            if result:
                plot_url = result.plot_image
            else:
                predicted_params = {}
                for param in ['a0', 'a1', 'n0', 'n1']:
                    value, _ = predict_parameter(param, voltage, temperature)
                    predicted_params[param] = value

                x = pd.read_csv("53L/100K53L/xchdata.csv", header=None)[0].values
                y_data = pd.read_csv("17L/all_charge.csv", header=None)
                y_column = y_data.iloc[:, 0].dropna().values
                min_len = min(len(x), len(y_column))
                x = x[:min_len]
                y_column = y_column[:min_len]
                y_pred = physical_model(
                    x,
                    predicted_params['a0'],
                    predicted_params['a1'],
                    predicted_params['n0'],
                    predicted_params['n1'],
                    tsi,
                    n_max
                )
                plot_url = generate_plot(x, y_column, y_pred)

                db_result = PredictionResult(
                    temperature=temperature,
                    voltage=voltage,
                    layers=layers,
                    n_max=n_max,
                    thickness=thickness,
                    plot_image=plot_url
                )
                db.session.add(db_result)
                db.session.commit()

        except Exception as e:
            return f"Error: {e}"
    previous_results = PredictionResult.query.order_by(PredictionResult.id.desc()).limit(10).all()

    return render_template("index.html", plot_url=plot_url, thickness=thickness, history=previous_results)

if __name__ == "__main__":
    app.run(debug=True)
