from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import os
from model import predict_parameter, physical_model  # Make sure model.py is in the same folder

app = Flask(__name__)

a = 5.43e-10  # Constant for thickness calculation
tsi = 2.17    # Given in model.py

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    thickness = None
    predicted_params = {}

    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            voltage = float(request.form["voltage"])
            layers = int(request.form["layers"])
            n_max = float(request.form["nmax"])

            # Calculate thickness
            thickness = (layers - 1) * (a / 4)

            # Predict parameters
            for param in ['a0', 'a1', 'n0', 'n1']:
                value, _ = predict_parameter(param, voltage, temperature)
                predicted_params[param] = value

            # Load xchdata.csv
            x = pd.read_csv("53L/100K53L/xchdata.csv", header=None)[0].values

            # Load y data from all_charge.csv (hardcoded column 0 for now)
            y_data = pd.read_csv("17L/all_charge.csv", header=None)
            y_column = y_data.iloc[:, 0].dropna().values

            # Ensure x and y same length
            min_len = min(len(x), len(y_column))
            x = x[:min_len]
            y_column = y_column[:min_len]

            # Predict y using physical model
            y_pred = physical_model(
                x,
                predicted_params['a0'],
                predicted_params['a1'],
                predicted_params['n0'],
                predicted_params['n1'],
                tsi,
                n_max
            )

            # Plot
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

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", plot_url=plot_url, thickness=thickness)

if __name__ == "__main__":
    app.run(debug=True)