# Car MSRP Prediction with Machine Learning

This project predicts the Manufacturer Suggested Retail Price (MSRP) of a car using machine learning.
this is the project link: https://the-car-price-predictor-repo-ngsbc3sefeiep5vq3ykazw.streamlit.app

The project covers the full machine learning workflow, including data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, error analysis, and deployment using Streamlit.

## Dataset

The dataset contains information about different cars, including:

- Make and model
- Year
- Engine horsepower
- Engine cylinders
- Transmission type
- Driven wheels
- Vehicle size and style
- City and highway MPG
- Popularity
- MSRP

The target variable is `MSRP`.

## Project Workflow

1. Loaded and inspected the dataset.
2. Checked missing values and duplicated rows.
3. Explored numerical and categorical features.
4. Created two new features:
   - Vehicle Age
   - Average MPG
5. Prepared numerical and categorical columns using:
   - SimpleImputer
   - MinMaxScaler
   - OneHotEncoder
   - ColumnTransformer
6. Compared Linear Regression and Decision Tree Regressor.
7. Tuned the Decision Tree using GridSearchCV and RandomizedSearchCV.
8. Evaluated the model using MAE, MSE, and RMSE.
9. Investigated prediction errors using residual analysis.
10. Deployed the trained model using Streamlit.

## Project Files

- `03_ML_reg_ex.ipynb`: Data analysis and model training notebook.
- `ml3.py`: Streamlit application.
- `model.pkl`: Saved machine learning model.
- `data.csv`: Car dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Run the Streamlit App

Install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

Make sure these files are located in the same folder:

```text
ml3.py
model.pkl
data.csv
```

Run the application:

```bash
streamlit run ml3.py
```

Enter the car information in the sidebar and press **Predict** to receive the predicted MSRP.

## Model Limitations

The model may produce larger errors for rare, luxury, or high-performance cars. Car prices may also depend on information that is not included in the dataset, such as mileage, vehicle condition, optional equipment, and market demand.

## Author

Data Science Bootcamp Project
