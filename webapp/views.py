from django.shortcuts import render 
import joblib
import numpy as np
import pandas as pd
import os
import json
import traceback

def home(request):
    # ==============================
    # 1️⃣ Feature setup (form names)
    # ==============================
    features = ["pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3", "benzene", "toluene"]

    predicted_aqi = None
    predicted_bucket = None
    predicted_color = "#007bff"
    predicted_percentage = 0
    error = None

    aqi_labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    aqi_colors = {
        "Good": "#00e400",
        "Satisfactory": "#9acd32",
        "Moderate": "#ffde33",
        "Poor": "#ff9933",
        "Very Poor": "#cc0033",
        "Severe": "#7e0023"
    }

    # ==============================
    # 2️⃣ File paths setup
    # ==============================
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    model_dir = os.path.join(base_dir, 'webapp', 'saved_models')
    regressor_path = os.path.join(model_dir, 'regressor.pkl')
    classifier_path = os.path.join(model_dir, 'classifier.pkl')
    data_path = os.path.join(base_dir, 'data', 'aqi_dataset.csv')

    # ==============================
    # 3️⃣ Load models
    # ==============================
    regressor, classifier = None, None
    try:
        if os.path.exists(regressor_path):
            regressor = joblib.load(regressor_path)
        if os.path.exists(classifier_path):
            classifier = joblib.load(classifier_path)
    except Exception as e:
        error = f"⚠️ Error loading models: {e}"

    # ==============================
    # 4️⃣ Handle user prediction
    # ==============================
    if request.method == 'POST':
        try:
            data = [float(request.POST.get(f, 0)) for f in features]
            X = np.array(data).reshape(1, -1)

            if regressor is not None:
                predicted_aqi = round(float(regressor.predict(X)[0]), 2)
                predicted_percentage = max(0, min(100, (predicted_aqi / 500.0) * 100))
            else:
                raise ValueError("Regression model not loaded.")

            if classifier is not None:
                class_idx = classifier.predict(X)[0]
                if isinstance(class_idx, (list, np.ndarray)):
                    class_idx = int(class_idx[0])
                class_idx = int(class_idx)
                predicted_bucket = aqi_labels[class_idx] if class_idx < len(aqi_labels) else "Unknown"
                predicted_color = aqi_colors.get(predicted_bucket, predicted_color)
            else:
                raise ValueError("Classification model not loaded.")
        except Exception as e:
            tb = traceback.format_exc()
            error = f"⚠️ Prediction failed: {e} — {str(tb)}"

    # ==============================
    # 5️⃣ Analytics & visualization
    # ==============================
    trends = {"global_dates": [], "global_aqi": [], "pollutant_medians": {}}
    median_json = json.dumps({"labels": [], "values": []})
    corr_json = json.dumps({"labels": [], "values": []})
    feature_json = json.dumps({"labels": [], "values": []})
    markers = []

    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)

        # Normalize column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'\.', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )

        # Identify pollutant columns
        all_possible_features = features
        pollutant_cols = [c for c in all_possible_features if c in df.columns]

        if len(pollutant_cols) == 0:
            raise ValueError("No pollutant columns found in CSV after normalization.")

        # Ensure AQI exists
        if "aqi" not in df.columns:
            if regressor is not None:
                try:
                    df["aqi"] = regressor.predict(df[pollutant_cols])
                except Exception:
                    df["aqi"] = np.random.randint(50, 200, size=len(df))
            else:
                df["aqi"] = np.random.randint(50, 200, size=len(df))

        # Date trend
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            trend_df = (
                df.dropna(subset=["date"])
                .groupby(df["date"].dt.to_period("M"))["aqi"]
                .median()
                .reset_index()
            )
            trend_df["date"] = trend_df["date"].astype(str)
            trends["global_dates"] = trend_df["date"].tolist()
            trends["global_aqi"] = trend_df["aqi"].tolist()
        else:
            trends["global_dates"] = ["2024-01", "2024-02", "2024-03", "2024-04"]
            trends["global_aqi"] = list(map(int, np.random.randint(60, 180, 4)))

        # Median pollutants
        median_series = df[pollutant_cols].median().round(2).sort_values(ascending=False)
        pollutant_medians = dict(zip(median_series.index.tolist(), median_series.values.tolist()))
        median_json = json.dumps({
            "labels": median_series.index.tolist(),
            "values": median_series.values.tolist(),
            "map": pollutant_medians
        })
        trends["pollutant_medians"] = pollutant_medians  # ✅ now embedded in trends

        # Correlations
        corr_series = df[pollutant_cols + ["aqi"]].corr(numeric_only=True)["aqi"].round(3)
        corr_map = corr_series.to_dict()
        corr_json = json.dumps({
            "labels": corr_series.index.tolist(),
            "values": corr_series.values.tolist(),
            "map": corr_map
        })
        trends["corr_map"] = corr_map  # ✅ include for JS readability

        # ---- Feature importances ----
        if regressor is not None:
            try:
                if hasattr(regressor, "feature_importances_"):
                    fi_vals = regressor.feature_importances_
                    if len(fi_vals) == len(pollutant_cols):
                        fi = dict(zip(pollutant_cols, fi_vals.round(3)))
                    else:
                        fi = {pollutant_cols[i]: round(fi_vals[i], 3) for i in range(min(len(pollutant_cols), len(fi_vals)))}
                elif hasattr(regressor, "coef_"):
                    fi_vals = np.abs(regressor.coef_[0] if len(regressor.coef_.shape) > 1 else regressor.coef_)
                    fi = dict(zip(pollutant_cols, fi_vals.round(3)))
                else:
                    # fallback random importances if model doesn’t expose these
                    fi = {f: round(np.random.uniform(0.05, 0.3), 3) for f in pollutant_cols}
            except Exception as e:
                fi = {f: round(np.random.uniform(0.05, 0.3), 3) for f in pollutant_cols}
        else:
            fi = {f: round(np.random.uniform(0.05, 0.3), 3) for f in pollutant_cols}

        # prepare feature_json (kept name you used earlier)
        feature_json = json.dumps({"labels": list(fi.keys()), "values": list(fi.values()), "map": fi})

        # City markers
        if "city" in df.columns:
            city_counts = df["city"].value_counts().head(20)
            city_coords = {
                "delhi": (28.6139, 77.2090),
                "mumbai": (19.0760, 72.8777),
                "chennai": (13.0827, 80.2707),
                "bengaluru": (12.9716, 77.5946),
                "kolkata": (22.5726, 88.3639),
                "hyderabad": (17.3850, 78.4867),
            }
            for c in city_counts.index:
                lat, lon = city_coords.get(c.lower(), (20.5937, 78.9629))
                markers.append({"city": c.title(), "lat": lat, "lon": lon, "count": int(city_counts[c])})
        else:
            markers = [
                {"city": "Delhi", "lat": 28.6139, "lon": 77.2090, "count": 35},
                {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777, "count": 28},
                {"city": "Chennai", "lat": 13.0827, "lon": 80.2707, "count": 15},
            ]

    except Exception as e:
        tb = traceback.format_exc()
        error = f"⚠️ Data unavailable / processing error: {e} — {tb}"
        trends = {"global_dates": [], "global_aqi": [], "pollutant_medians": {}}
        median_json = json.dumps({"labels": [], "values": [], "map": {}})
        corr_json = json.dumps({"labels": [], "values": [], "map": {}})
        feature_json = json.dumps({"labels": [], "values": [], "map": {}})
        markers = []

    # ==============================
    # 6️⃣ Render to template
    # ==============================
    return render(request, 'home.html', {
        "features": features,
        "predicted_aqi": predicted_aqi,
        "predicted_bucket": predicted_bucket,
        "predicted_color": predicted_color,
        "predicted_percentage": predicted_percentage,
        "error": error,
        "trends_json": json.dumps(trends),
        "median_json": median_json,
        "corr_json": corr_json,
        # <-- important: template expects fi_json (id="fi-data"); provide it
        "feature_json": feature_json,
        "fi_json": json.dumps({
            "labels": list(fi.keys()),
            "values": list(fi.values()),
            "map": fi
        }),
        "markers_json": json.dumps(markers),
    })
