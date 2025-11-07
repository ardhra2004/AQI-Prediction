from sklearn.preprocessing import LabelEncoder

def select_features(df):
    """
    Selects feature columns for AQI prediction (regression/classification).

    Returns:
        X (pd.DataFrame): Feature matrix
        y_reg (pd.Series): Target for regression (AQI)
        y_clf (np.array or None): Encoded target for classification (AQI_Bucket)
    """

    feature_cols = [
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
        'CO', 'SO2', 'O3', 'Benzene', 'Toluene'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]  

    X = df[feature_cols].copy()

    # Regression target
    y_reg = df['AQI']

    # Classification target
    if 'AQI_Bucket' in df.columns:
        le = LabelEncoder()
        y_clf = le.fit_transform(df['AQI_Bucket'])
    else:
        y_clf = None

    print(f"Selected {len(feature_cols)} features: {feature_cols}")
    print(f"Regression target: 'AQI'")
    if y_clf is not None:
        print(f"Classification target: 'AQI_Bucket' (encoded)")
    else:
        print("Classification target not found!")

    return X, y_reg, y_clf
