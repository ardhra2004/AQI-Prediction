import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path: str):
    """
    Loads and cleans the AQI dataset.
    - Removes duplicates
    - Drops unnecessary or null-filled columns
    - Handles missing values
    - Encodes AQI_Bucket into numeric labels
    - Converts date columns
    """

    print("\n Loading and cleaning dataset...")

    df = pd.read_csv(file_path)
    print(f"🔹 Original Dataset Shape: {df.shape}")
    print(f"🔹 Columns: {list(df.columns)}")

    df.drop_duplicates(inplace=True)

    df.dropna(subset=['AQI', 'AQI_Bucket'], inplace=True)

    if 'Xylene' in df.columns:
        df.drop(columns=['Xylene'], inplace=True)
        

    df.dropna(inplace=True)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    le = LabelEncoder()
    df['AQI_Class'] = le.fit_transform(df['AQI_Bucket'])

    print(f"✅ Cleaned Dataset Shape: {df.shape}")
    print(f"✅ AQI Classes: {list(le.classes_)}")

    return df, le