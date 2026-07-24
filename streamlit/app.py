import streamlit as st

from prediction_utils import (
    create_input_dataframe,
    load_model,
    load_preprocessor,
    predict_weekly_sales,
)


# Page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Walmart Weekly Sales Prediction",
    page_icon="🛒",
    layout="wide",
)
st.sidebar.title("Project Information")

st.sidebar.success(
    "Machine Learning Regression Project"
)

st.sidebar.markdown("""
**Model**
- Linear Regression

**Dataset**
- Walmart Weekly Sales

**Target**
- Weekly Sales

**Framework**
- Streamlit
""")


@st.cache_resource
def load_prediction_resources():
    """
    Load and cache the saved preprocessor and model.
    """

    preprocessor = load_preprocessor()
    model = load_model()

    return preprocessor, model


# Application heading.
st.title("🛒 Walmart Weekly Sales Prediction System")

st.markdown(
    """
This application predicts **Walmart Weekly Sales**
using a trained **Linear Regression Machine Learning Model**.

Simply enter the store information,
economic indicators and calendar information,
then click **Predict Weekly Sales**.
"""
)


# Load the saved objects.
try:
    preprocessor, model = load_prediction_resources()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(
        f"Unable to load the prediction resources: {error}"
    )
    st.stop()


st.success("Prediction model loaded successfully.")

st.info("""
Model Used: Linear Regression

The predicted weekly sales are estimated using the trained machine learning model.
""")


# Prediction form.
with st.form("sales_prediction_form"):

    st.subheader("Store Information")

    store = st.selectbox(
        "Store number",
        options=list(range(1, 46)),
    )

    holiday_option = st.selectbox(
        "Is this a holiday week?",
        options=[
            "No",
            "Yes",
        ],
    )

    holiday_flag = (
        1 if holiday_option == "Yes" else 0
    )


    st.subheader("Economic Information")

    column_1, column_2 = st.columns(2)

    with column_1:

        temperature = st.number_input(
            "Temperature",
            min_value=-20.0,
            max_value=120.0,
            value=60.0,
            step=0.1,
        )

        fuel_price = st.number_input(
            "Fuel price",
            min_value=0.0,
            value=3.00,
            step=0.01,
        )

    with column_2:

        cpi = st.number_input(
            "Consumer Price Index (CPI)",
            min_value=0.0,
            value=210.0,
            step=0.1,
        )

        unemployment = st.number_input(
            "Unemployment rate",
            min_value=0.0,
            value=8.0,
            step=0.1,
        )


    st.subheader("Calendar Information")

    column_3, column_4 = st.columns(2)

    with column_3:

        year = st.selectbox(
            "Year",
            options=[
                2010,
                2011,
                2012,
            ],
        )

        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
        )

        quarter = st.selectbox(
            "Quarter",
            options=[
                1,
                2,
                3,
                4,
            ],
        )

    with column_4:

        week_of_year = st.number_input(
            "Week of year",
            min_value=1,
            max_value=53,
            value=1,
            step=1,
        )

        time_index = st.number_input(
            "Time index",
            min_value=0,
            value=0,
            step=1,
        )


    submitted = st.form_submit_button(
        "Predict Weekly Sales",
        use_container_width=True,
    )


# Run prediction only after the button is pressed.
if submitted:

    input_data = create_input_dataframe(
        store=store,
        holiday_flag=holiday_flag,
        temperature=temperature,
        fuel_price=fuel_price,
        cpi=cpi,
        unemployment=unemployment,
        year=year,
        month=month,
        week_of_year=week_of_year,
        quarter=quarter,
        time_index=time_index,
    )

    st.subheader("Input Preview")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True,
    )

    try:
        predicted_sales = predict_weekly_sales(
            preprocessor=preprocessor,
            model=model,
            input_data=input_data,
        )

        st.subheader("Prediction Result")
        st.info(
            "Prediction generated using the trained Linear Regression model."
        )

        st.metric(
    label="💰 Predicted Weekly Sales",
    value=f"${predicted_sales:,.2f}",

        )

        st.success(
            "The weekly-sales prediction was completed successfully."
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
    st.markdown("---")

    st.caption(
        "Developed by Udochi Tonia Madu • Walmart Weekly Sales Prediction • Streamlit + Scikit-learn"
        )