# Project Documentation — Life Expectancy Prediction (ML Regression)

## How to Run the Analysis
### Run in Google Colab
1. Upload:
   AI_ML_GROUP_PROJECT.ipynb
   Life Expectancy Dataset.csv
2. Open the notebook and run all cells in order.

### Run Locally (Anaconda / VS Code / Jupyter)
1. Ensure Python 3 is installed (recommended: Anaconda).
2. Install dependencies:

## Project Overview
This project predicts Life Expectancy using country-level socioeconomic and health indicators. The workflow follows a standard machine learning pipeline:
1. Data loading and inspection  
2. Exploratory Data Analysis (EDA)
3. Missing value handling and data cleaning
4. Feature engineering
5. Categorical encoding (one-hot)
6. Train/test split and scaling
7. Model development and evaluation (with k-fold CV)
8. Feature importance (XGBoost)
9. Hyperparameter tuning (RandomizedSearchCV)
10. Results visualisation (tables and plots)

**The goal is to build and compare multiple regression models and interpret which factors most strongly influence life expectancy.**

The dataset contains demographic, health, and economic indicators across multiple years and countries/regions. Example variables include:  
Adult_mortality, Infant_deaths, Under_five_deaths, GDP_per_capita, Hepatitis_B, Polio, Diphtheria, Schooling, Population_mln.  
Target variable is **Life_expectancy**

## Import and Setup
Libraries used:  
- numpy, pandas: data manipulation and numerical operations  
- matplotlib, seaborn: charts and plots for EDA and results  
- scipy.stats.skew: skewness calculation (e.g., population distribution)  
- sklearn: modelling, preprocessing, evaluation, feature selection, CV, tuning  
- xgboost: gradient boosting regression model

## Data Loading and Initial Inspection
- Key checks: data.head(), data.columns, data.info(), data.describe()  
- Missing values summary with data.isnull().sum()  
Why this matters:  
- Before any cleaning or modelling, we need to confirm which variables are numeric vs categorical, where missing values exist, typical ranges of values

## Exploratory Data Analysis (EDA)
- Distribution plot of Life_expectancy  
- Correlation matrix heatmap (numeric features)  
- Scatter plots of GDP_per_capita vs Life_expectancy, Schooling vs Life_expectancy, Adult_mortality vs Life_expectancy (plus regression line)  
- Boxplot of Life_expectancy by region.  
- Line plot of average Life_expectancy over time.  
- Region trend lines over time.

**Why this was done:**  
Exploratory Data Analysis provides evidence for which features correlate strongly with the target whether relationships appear linear or non-linear, potential outliers and skewed features, time trends and regional differences.  
These results inform later decisions (e.g., feature engineering, model choices).

## Data Cleaning  
**Handling Missing Values (Mean Imputation for Selected Columns)**  
- The code fills missing values for GDP_per_capita, Hepatitis_B, Polio using column means.
```python
#Handling null values
data[["GDP_per_capita", "Hepatitis_B", "Polio"]] = (
    data[["GDP_per_capita", "Hepatitis_B", "Polio"]]
    .fillna(data[["GDP_per_capita", "Hepatitis_B", "Polio"]].mean())
)
print(data.isnull().sum())
```
Why mean imputation was used because:
- It maintains the dataset size (avoids losing rows).  
- It is simple and acceptable for baseline modelling.
- It is appropriate for variables where missingness is not extremely high.

**Outlier Detection (IQR Method)**
The code calculates:  
- Q1, Q3, IQR, lower/upper bounds  
- Outlier mask and counts per column  
- Rows containing at least one outlier
```python
  #Select only numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
print("Numeric columns:")
print(list(numeric_cols))

#Compute IQR, lower and upper bounds for each numeric column
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR  
```

**Why this approach**:
- IQR method is robust and widely used for numeric outlier detection.  
- We used this primarily to inspect outliers, not automatically delete large amounts of data.

**Data Cleaning**  
A cleaned dataset (clean_data) is created from data.copy().  
- Remove negative values in numeric columns (keeps only >= 0)  
- Remove impossible values: BMI must be > 0  
- Life expectancy must be between 20 and 120  
- Schooling must be between 0 and 25  
- GDP must be > 0

## Feature Engineering
A prepared dataset (data_prep) is created from clean_data.copy().
Engineered features:
**Thinness_avg** was created by taking the average of Thinness_ten_nineteen_years and Thinness_five_nine_years
```python
Thinness_avg = (Thinness_ten_nineteen_years + Thinness_five_nine_years) / 2
```
Since they are highly correlated with one another and it reduces redundancy and creates a single malnutrition indicator.

**Vaccination index** was created by taking the average of Polio, Hepatitis_B, Diphtheria values.  
```python
Vaccination_index = (Polio + Hepatitis_B + Diphtheria) / 3
```  
It was done because it summarises vaccination coverage, it is easier to interpret as a health system strength proxy.

**Log population**. The log og the population was taken to reduce the skewness.
```python
Log_population = log1p(Population_mln)
```
Population is highly skewed, log transform reduces skewness and helps some models.

**Total child deaths** was created by combining the Infant_deaths and Under_five_deaths values as one
```python
Total_child_deaths = Infant_deaths + Under_five_deaths
```
The reason is that it captures overall child mortality burden.

**Then the original variables replaced by engineered features are dropped (e.g., thinness components, vaccination components, infant/under-five deaths, population).**

## Encoding Categorical Variables (One-Hot Encoding)
- Country and Region are converted using:
```python
categorical_cols = ['Country', 'Region']
data_prep_encoded = pd.get_dummies(
    data_prep,
    columns=['Country', 'Region'],
    drop_first=True,
    dtype=int
)
data_prep_encoded.head()
```

Why one-hot encoding:
Country and Region are nominal (no inherent order).
Label encoding would introduce artificial ranking (e.g., Country 1 < Country 2), which is incorrect.
Why drop_first=True:
```python
drop_first=True,
```
Drops one category per variable to avoid the dummy variable trap (perfect multicollinearity in linear models).

Why dtype=int:
```python
dtype=int
```
Converts dummy columns to numeric 0/1, ensuring compatibility across all models and metrics.

## Defining X and y, and Train/Test Split  
y = Life_expectancy  
X = all other columns  
Split: 80% train / 20% test with random_state=42
```python
#Define target (y) and features (X)
y = data_prep_encoded["Life_expectancy"]                 # target variable
X = data_prep_encoded.drop("Life_expectancy", axis=1)    # all other columns

print("X shape before split:", X.shape)
print("y shape before split:", y.shape)

#Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42)     # reproducibility
```
Why split: The test set simulates unseen data and provides a more honest estimate of generalisation.

## Scaling
The notebook includes manual scaling using StandardScaler to create:
X_train_scaled, X_test_scaled  
```python
# Pipelines for models that REQUIRE scaling
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR())
])
```

**Important note**: Scaling is performed inside pipelines for scale-sensitive models (LR, SVR) rather than manually transforming the dataset, which prevents data leakage during cross-validation.  

## Model Development and Cross-Validation (K-Fold)
Models implemented:
Linear Regression (pipeline with scaling)
SVR (pipeline with scaling)
Random Forest
Decision Tree
XGBoost
```python
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=-1),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())  # default kernel='rbf'
    ]),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    )
}
```

Cross-validation uses:
KFold(n_splits=5, shuffle=True, random_state=42)
cross_validate(...) with scoring for MAE, MSE (converted to RMSE), R², and MAPE
```python
def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_scorer = make_scorer(mape_score, greater_is_better=False)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
    "mape": mape_scorer
}
results = []
for name, model in models.items():
    cv = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    mae  = -cv["test_mae"].mean()
    rmse = np.sqrt(-cv["test_mse"].mean())
    r2   = cv["test_r2"].mean()
    mape = -cv["test_mape"].mean()
    results.append([name, rmse, mae, mape, r2])

comparison_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "MAPE (%)", "R²"]).sort_values("RMSE")
comparison_df
```
**Why k-fold CV?**  
- Reduces dependence on a single train/test split.  
- Provides more reliable performance estimates (average over folds).  
- Required by the coursework brief

## Results Summary and Visualisation
Outputs produced:
Model comparison table: RMSE, MAE, MAPE, R²  
```python
Running 5-fold cross-validation for each model...

MODEL PERFORMANCE COMPARISON (5-fold CV averages):
               Model      RMSE       MAE  MAPE (%)        R²
4            XGBoost  0.451512  0.323678  0.498777  0.997704
0  Linear Regression  0.509008  0.345519  0.540532  0.997063
1      Random Forest  0.541542  0.362687  0.566290  0.996696
3      Decision Tree  0.791188  0.528190  0.824800  0.992935
2                SVR  1.515280  0.819598  1.413615  0.973971
```
Combined bar + line chart (errors as bars, R² as line)

Multiple metrics were used because:
RMSE penalises large errors
MAE is robust and interpretable
MAPE provides relative % error
R² shows variance explained

## Feature Importance (XGBoost)
Feature selection for XGBoost uses built-in model importance.
It trains XGBoost (after optionally dropping country/region dummies)
Extracts feature_importances_
Displays top 20 features and plots them  
```python
#FEATURE SELECTION (XGBoost Importance)
xgb_importance_df = pd.DataFrame({
    "Feature": X_xgb_train.columns,
    "Importance": final_xgb.feature_importances_
}).sort_values("Importance", ascending=False)

top_n = 20
display(xgb_importance_df.head(top_n))

Feature	Importance
13	Total_child_deaths	0.525248
1	Adult_mortality	0.177638
8	Economy_status_Developed	0.120532
5	Incidents_HIV	0.098752
7	Schooling	0.031193
6	GDP_per_capita	0.020899
11	Vaccination_index	0.005193
3	Measles	0.004386
10	Thinness_avg	0.004322
4	BMI	0.004277
2	Alcohol_consumption	0.003539
12	Log_population	0.002350
0	Year	0.001027
9	Economy_status_Developing	0.000643
```

**Why this method**
XGBoost provides an embedded, model-based feature importance measure.
It is suitable for non-linear tree ensembles and interpretable for presentation/reporting.

**Why drop Country/Region for XGBoost importance**
- Country and region dummies can dominate importance due to strong proxy effects.
- Dropping them helps focus interpretation on socioeconomic/health drivers (aligned with project objective).

## Hyperparameter Tuning (RandomizedSearchCV)
RandomizedSearchCV is used with:
cv=5 (5-fold cross-validation inside tuning)
scoring based on RMSE (neg_root_mean_squared_error)
parameter distributions for RF, SVR, DT, XGBoost  
```python  
RANDOMIZED SEARCH RESULTS (Test Set Evaluation):
Model	RMSE	MAE	MAPE (%)	R²
4	XGBoost	0.349178	0.253473	0.389931	0.998531
2	Support Vector Regression	0.415613	0.213715	0.340200	0.997919
0	Linear Regression	0.493560	0.339952	0.531663	0.997065
1	Random Forest	0.510063	0.364487	0.567105	0.996865
3	Decision Tree	0.768887	0.526527	0.823224	0.992877  
```

**Why RandomizedSearchCV**
- More efficient than GridSearch for large search spaces (especially XGBoost)
- Achieves strong tuning results with fewer evaluations
- Final tuned models are evaluated on the test set and plotted.
## Conclusion  
Using the Life Expectancy dataset, the prediction model was built. The best model with a RMSE score of 0.34 years (4 months) and a R-squared score of 99.8% is XGBoost.

## Limitations and Areas for Improvement
- Mean imputation may bias results if missingness is systematic.
- Extremely high R² values may indicate strong feature redundancy or country proxy effects.
- Country/Region can inflate accuracy. Results should be interpreted with care.

## Future improvements
Evaluate generalisation by leaving out entire countries (grouped CV)
Use SHAP values for deeper model explainability (especially XGBoost)
