import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def preprocess_data(dataset, target_column, save_path, file_path):
    # Drop Fitur yang Tidak Relevan
    clean_df = dataset.copy()
    clean_df = clean_df.drop(columns="Region")

    # Pendefinisian Fitur Numerik dan Kategorik
    numerical_features = [
        col for col in clean_df.select_dtypes(include=['float64', 'int64']).columns
        if col != target_column
    ]
    categorical_features = clean_df.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Penghapusan Outlier
    for col in numerical_features:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Hapus baris yang nilai kolomnya di luar batas IQR
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]

    # Mendapatkan nama kolom tanpa kolom target
    column_names = clean_df.columns.drop(target_column)
 
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
 
    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Pipeline untuk Fitur Kategorik
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Pipeline untuk Fitur Numerik
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("escaler", StandardScaler())
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Data Splitting
    X = clean_df.drop(columns="Yield_tons_per_hectare")
    Y = clean_df["Yield_tons_per_hectare"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Fitting dan Transformasi pada Training Set
    X_train = preprocessor.fit_transform(X_train)

    # Fitting dan Transformasi pada Testing Set
    X_test = preprocessor.transform(X_test)

    # Standardisasi pada Target
    Y_scaler = StandardScaler()
    Y_train = pd.DataFrame(
        Y_scaler.fit_transform(Y_train.values.reshape(-1, 1)),
        columns=[Y_train.name],
        index=Y_train.index
    )
    Y_test = pd.DataFrame(
        Y_scaler.transform(Y_test.values.reshape(-1, 1)),
        columns=[Y_test.name],
        index=Y_test.index
    )

    # Ubah X menjadi dataframe
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # Simpan pipeline
    joblib.dump(preprocessor, save_path)

    # Simpan data preprocessed
    os.makedirs("preprocessing/dataset", exist_ok=True)
    X_train.to_csv("preprocessing/dataset/X_train.csv", index=False)
    X_test.to_csv("preprocessing/dataset/X_test.csv", index=False)
    Y_train.to_csv("preprocessing/dataset/Y_train.csv", index=False)
    Y_test.to_csv("preprocessing/dataset/Y_test.csv", index=False)

    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    dataset = pd.read_csv("crop_yield.csv")
    preprocess_data(dataset, "Yield_tons_per_hectare", "preprocessing/preprocessor_pipeline.joblib", "preprocessing/dataset.csv")