# NYC-Airbnb-Room-Type-Prediction
NYC Airbnb Room Type Prediction — A scikit-learn machine-learning project that predicts an Airbnb listing’s room type from listing details, with a Streamlit app for interactive predictions.

# NYC Airbnb Room Type Prediction

A machine-learning project that predicts the room type of a New York City Airbnb listing using listing details such as location, price, minimum nights, reviews, and availability.

## Project workflow

The Jupyter notebook:

1. Downloads and explores the NYC Airbnb Open Data dataset.
2. Cleans the data, handles missing review values, and caps extreme price and minimum-night values.
3. Splits the data into training, validation, and test sets.
4. Builds preprocessing pipelines for numeric and categorical features.
5. Compares Logistic Regression, Random Forest, XGBoost, and HistGradientBoosting models using cross-validation and validation metrics.
6. Tunes an XGBoost model with randomized search.
7. Saves the selected model and its target class names to `final_airbnb_model.pkl`.

The target classes are `Entire home/apt`, `Private room`, and `Shared room`.

## Streamlit app

The app collects listing details and uses the saved model to display a predicted room type and class probabilities.

## Setup

Use Python 3.9 or later. In a terminal, install the dependencies:

```bash
pip install pandas numpy scikit-learn xgboost kagglehub matplotlib streamlit
```

Run the notebook to download the dataset and create `final_airbnb_model.pkl`. Then start the app:

```bash
streamlit run app.py
```

## Dataset

The notebook downloads the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) dataset using `kagglehub`.

## Notes

- The app requires `final_airbnb_model.pkl` in the same folder as `app.py`.
- Only load pickle files from sources you trust.
- The notebook creates a test split, but the reported model comparisons and tuning use cross-validation and validation data.
