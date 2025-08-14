import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --------------------------------------------------------------------
# 1) Register ALL your (layer, temperature) directories here
#    (paste the list you shared; no edits elsewhere needed)
# --------------------------------------------------------------------
DATA_DIRECTORIES = [
    r"C:\Users\acer\Desktop\web\n1a1data\17L\15K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\50K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\77K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\100K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\150K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\200K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\250K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\300K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\15K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\50K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\77K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\100K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\150K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\200K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\250K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\300K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\15K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\50K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\77K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\100K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\150K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\200K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\250K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\300K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\15K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\50K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\77K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\100K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\150K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\200K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\250K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\300K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\15K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\50K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\77K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\100K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\150K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\200K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\250K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\300K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\15K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\50K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\77K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\100K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\150K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\200K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\250K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\300K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\15K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\50K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\77K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\100K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\150K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\200K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\250K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\300K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\15K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\50K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\77K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\100K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\150K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\200K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\250K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\300K75L",
]

# Build a lookup: (layers:int, temperature:int) -> absolute folder path
def parse_layers_temp_from_path(folder_path: str):
    """
    Expects folders ending with '<temp>K<layers>L', e.g. ...\\23L\\300K23L
    Returns (layers, temperature) as ints.
    """
    base = os.path.basename(os.path.normpath(folder_path))  # e.g. "300K23L"
    # split at 'K' -> ['300', '23L']
    t_str, l_str = base.split('K')
    temperature = int(t_str)
    layers = int(l_str[:-1])  # remove trailing 'L'
    return layers, temperature

DIRECTORY_MAP = {}
for p in DATA_DIRECTORIES:
    try:
        L, T = parse_layers_temp_from_path(p)
        DIRECTORY_MAP[(L, T)] = p
    except Exception:
        # silently ignore paths that don't match the pattern
        pass

# ------------------------------------------
# Physics helpers
# ------------------------------------------
def calculate_tsi_and_positions(num_layers: int):
    """
    TSI in nm; x positions symmetrical across thickness center.
    Step per layer center ~ 0.13575 nm (as in your previous code).
    """
    step = 0.13575
    tsi = (num_layers - 1) * step  # total thickness approximation
    # If you prefer x directly from file, you can read from xchdata.csv.
    # We'll still default to file x below if available.
    half_points = num_layers // 2
    xs = []
    if num_layers % 2 == 1:
        for i in range(half_points, 0, -1):
            xs.append(-i * step)
        xs.append(0.0)
        for i in range(1, half_points + 1):
            xs.append(i * step)
    else:
        for i in range(half_points, 0, -1):
            xs.append(-i * step + step/2)
        for i in range(half_points):
            xs.append((i + 0.5) * step)
    return tsi, np.array(xs, dtype=float)

def physical_model(x, a0, a1, n0, n1, tsi, n_max):
    """
    n(x) = n_max * ( n0 + n1*sinh^2(a1*pi*x/tsi) ) * cos^2(a0*pi*x/tsi)
    """
    if tsi == 0:
        return np.zeros_like(x)
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
    return n_max * (n0 + n1 * sinh2_term) * cos2_term

# ------------------------------------------
# Data loading / interpolation
# ------------------------------------------
def load_parameters_for(voltage: float, layers: int, temperature: int, folder: str):
    """
    Read dataprep.csv with columns: z, a1, n0, n1, a0, a0_avg
    Interpolate each column vs z at the given 'voltage'.
    a0 is gate-independent in physics, but we still read it (it will be constant across z).
    """
    dp_path = os.path.join(folder, "dataprep.csv")
    if not os.path.exists(dp_path):
        raise FileNotFoundError(f"Missing dataprep.csv in {folder}")

    df = pd.read_csv(dp_path)
    # make sure column names match exactly your files:
    # z,a1,n0,n1,a0,a0_avg
    required = ["z", "a1", "n0", "n1", "a0"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {dp_path}. Found: {list(df.columns)}")

    # sort by z just in case, drop NA
    df = df[required].dropna().sort_values("z")
    z_vals = df["z"].values.astype(float)

    def interp(col):
        vals = df[col].values.astype(float)
        # extrapolate flat at ends if voltage outside z range
        if voltage <= z_vals[0]:
            return float(vals[0])
        if voltage >= z_vals[-1]:
            return float(vals[-1])
        return float(np.interp(voltage, z_vals, vals))

    a0 = interp("a0")
    a1 = interp("a1")
    n0 = interp("n0")
    n1 = interp("n1")

    return a0, a1, n0, n1

def load_x_and_nmax(folder: str):
    """
    Use xchdata.csv for x; n_max is the maximum across all data columns (excluding x).
    If xchdata.csv missing, fallback to synthetic x from layers.
    """
    x_path = os.path.join(folder, "xchdata.csv")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing xchdata.csv in {folder}")

    df = pd.read_csv(x_path, header=None)
    if df.shape[1] == 0:
        raise ValueError(f"xchdata.csv in {folder} is empty")

    x = df.iloc[:, 0].astype(float).values
    if df.shape[1] > 1:
        y_vals = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
        n_max = np.nanmax(y_vals)
    else:
        # only x column present; set a safe default n_max=1
        n_max = 1.0

    if not np.isfinite(n_max):
        n_max = 1.0

    return x, float(n_max)

def make_plot(x, y, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Carrier Concentration n(x)")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------------------------
# Routes
# ------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # Fill dropdowns with available layers and temperatures from your directories
    layers_sorted = sorted(set(L for (L, T) in DIRECTORY_MAP.keys()))
    temps_sorted = sorted(set(T for (L, T) in DIRECTORY_MAP.keys()))
    return render_template("index.html", layers=layers_sorted, temperatures=temps_sorted)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        temperature = int(float(request.form["temperature"]))
        voltage = float(request.form["voltage"])
        layers = int(float(request.form["layers"]))
    except Exception:
        return jsonify({"error": "Please supply valid numeric Temperature, Voltage, and Layers."}), 400

    key = (layers, temperature)
    folder = DIRECTORY_MAP.get(key)
    if not folder:
        return jsonify({"error": f"No data folder for {layers}L at {temperature}K."}), 404

    # Load parameters from dataprep.csv (interpolated at voltage)
    try:
        a0, a1, n0, n1 = load_parameters_for(voltage, layers, temperature, folder)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Load x and n_max from xchdata.csv
    try:
        x, n_max = load_x_and_nmax(folder)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Compute tsi (for the physics expression); if you prefer file x range, keep tsi from layers
    tsi, _ = calculate_tsi_and_positions(layers)

    # Compute model prediction n(x)
    y_pred = physical_model(x, a0, a1, n0, n1, tsi, n_max)
    main_graph_b64 = make_plot(x, y_pred, f"Carrier Concentration @ {temperature}K, {layers}L, V={voltage}")

    # Build "past data" panels for all other folders, at the SAME user voltage
    past_graphs = []
    for (L, T), fold in DIRECTORY_MAP.items():
        if (L, T) == key:
            continue
        try:
            pa0, pa1, pn0, pn1 = load_parameters_for(voltage, L, T, fold)
            px, pnmax = load_x_and_nmax(fold)
            ptsi, _ = calculate_tsi_and_positions(L)
            py = physical_model(px, pa0, pa1, pn0, pn1, ptsi, pnmax)
            g = make_plot(px, py, f"{T}K, {L}L (V={voltage})")
            past_graphs.append({
                "temp": T,
                "layers": L,
                "n_max": float(pnmax),
                "a0": float(pa0),
                "a1": float(pa1),
                "n0": float(pn0),
                "n1": float(pn1),
                "graph": g
            })
        except Exception:
            # Skip misconfigured folders quietly
            continue

    # Sort past graphs nicely (by temperature, then layers)
    past_graphs.sort(key=lambda d: (d["temp"], d["layers"]))

    result = {
        "params": {
            "temperature": temperature,
            "layers": layers,
            "voltage": voltage,
            "a0": float(a0),
            "a1": float(a1),
            "n0": float(n0),
            "n1": float(n1),
            "n_max": float(n_max),
            "tsi": float(tsi),
        },
        "main_graph": main_graph_b64,
        "past_graphs": past_graphs[:12]  # cap to avoid huge payloads; adjust if you like
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
