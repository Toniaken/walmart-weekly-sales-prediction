from pathlib import Path
import pickle

import pandas as pd


# Find the project root folder.
CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_FOLDER = CURRENT_FILE.parent

PROJECT_ROOT = STREAMLIT_FOLDER.parent


# Paths to the saved preprocessing and model files.
PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "walmart_preprocessor.pkl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_walmart_sales_model.pkl"
)


def load_preprocessor():
    """
    Load the fitted Walmart data preprocessor.
    """

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    with open(PREPROCESSOR_PATH, "rb") as file:
        preprocessor = pickle.load(file)

    return preprocessor


def load_model():
    """
    Load the final trained regression model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


def create_input_dataframe(
    store,
    holiday_flag,
    temperature,
    fuel_price,
    cpi,
    unemployment,
    year,
    month,
    week_of_year,
    quarter,
    time_index,
):
    """
    Convert values entered by the user into a one-row DataFrame.
    """

    input_data = pd.DataFrame(
        {
            "store": [store],
            "holiday_flag": [holiday_flag],
            "temperature": [temperature],
            "fuel_price": [fuel_price],
            "cpi": [cpi],
            "unemployment": [unemployment],
            "year": [year],
            "month": [month],
            "week_of_year": [week_of_year],
            "quarter": [quarter],
            "time_index": [time_index],
        }
    )

    return input_data


def predict_weekly_sales(
    preprocessor,
    model,
    input_data,
):
    """
    Transform the input data and predict weekly sales.
    """

    prepared_input = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        prepared_input
    )

    return float(prediction[0])