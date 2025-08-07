# models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float, nullable=False)
    voltage = db.Column(db.Float, nullable=False)
    layers = db.Column(db.Integer, nullable=False)
    n_max = db.Column(db.Float, nullable=False)
    thickness = db.Column(db.Float, nullable=False)
    plot_image = db.Column(db.Text, nullable=False)
    date_searched = db.Column(db.DateTime, server_default=db.func.now())
