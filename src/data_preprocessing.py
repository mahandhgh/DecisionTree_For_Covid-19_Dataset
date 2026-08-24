import numpy as np
import pandas as pd


def load_and_prepare_data(file_path="Covid Data.csv"):
    covid_data = pd.read_csv(file_path)

    excluded_columns = ['DATE_DIED', 'INTUBED', 'ICU']
    covid_data_cleaned = covid_data.drop(columns=excluded_columns)

    missing_value_cols = [
        'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
        'OBESITY', 'RENAL_CHRONIC', 'TOBACCO'
    ]

    covid_data_cleaned[missing_value_cols] = (
        covid_data_cleaned[missing_value_cols]
        .replace({97: None, 99: None})
    )

    covid_data_cleaned.loc[
        covid_data_cleaned['SEX'] == 2, 'PREGNANT'
    ] = 2

    covid_data_cleaned['CLASIFFICATION_FINAL'] = (
        covid_data_cleaned['CLASIFFICATION_FINAL'] <= 3
    ).astype(int)

    categorical_cols = [
        'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
        'OBESITY', 'RENAL_CHRONIC', 'TOBACCO'
    ]

    covid_data_cleaned[categorical_cols] = (covid_data_cleaned[categorical_cols].apply(pd.to_numeric, errors='coerce'))

    covid_data_cleaned = covid_data_cleaned.dropna(subset=['CLASIFFICATION_FINAL'])

    X = covid_data_cleaned.drop(columns=['CLASIFFICATION_FINAL']).to_numpy()

    y = covid_data_cleaned['CLASIFFICATION_FINAL'].to_numpy()

    return covid_data_cleaned, X, y
