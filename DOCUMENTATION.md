# Project Documentation — Life Expectancy Prediction (ML Regression)

## How to Run the Analysis
## Run in Google Colab
1. Upload:
   AI_ML_GROUP_PROJECT.ipynb
   Life Expectancy Dataset.csv
2. Open the notebook and run all cells in order.

### Run Locally (Anaconda / VS Code / Jupyter)
1. Ensure Python 3 is installed (recommended: Anaconda).
2. Install dependencies:
   ```bash

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
Adult_mortality, Infant_deaths, Under_five_deaths, GDP_per_capita, Hepatitis_B, Polio, Diphtheria, Schooling, Population_mln`
Target variable is **Life_expectancy**
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost

**Imports and Setup**
Libraries used: numpy, pandas: data manipulation and numerical operations
matplotlib, seaborn: charts and plots for EDA and results
scipy.stats.skew: skewness calculation (e.g., population distribution)
sklearn: modelling, preprocessing, evaluation, feature selection, CV, tuning
xgboost: gradient boosting regression model

**Data Loading and Initial Inspection**
Key checks: data.head(), data.columns, data.info(), data.describe()
Missing values summary with data.isnull().sum()
Why this matters:
Before any cleaning or modelling, we need to confirm which variables are numeric vs categorical, where missing values exist, typical ranges of values

**Exploratory Data Analysis (EDA)**
Distribution plot of Life_expectancy
Correlation matrix heatmap (numeric features)
Scatter plots of GDP_per_capita vs Life_expectancy, Schooling vs Life_expectancy, Adult_mortality vs Life_expectancy (plus regression line)
Boxplot of Life_expectancy by region.
Line plot of average Life_expectancy over time.
Region trend lines over time.

**Why this was done:**
Exploratory Data Analysis provides evidence for which features correlate strongly with the target whether relationships appear linear or non-linear, potential outliers and skewed features, time trends and regional differences.
These results inform later decisions (e.g., feature engineering, model choices).

**Handling Missing Values (Mean Imputation for Selected Columns)**
The code fills missing values for GDP_per_capita, Hepatitis_B, Polio using column means.
Why mean imputation was used because it maintains the dataset size (avoids losing rows).
It is simple and acceptable for baseline modelling.
It is appropriate for variables where missingness is not extremely high.

**Outlier Detection (IQR Method)**
The code calculates:
Q1, Q3, IQR, lower/upper bounds
Outlier mask and counts per column
Rows containing at least one outlier

**Why this approach**:
IQR method is robust and widely used for numeric outlier detection.
We used this primarily to inspect outliers, not automatically delete large amounts of data.

**Data Cleaning**
A cleaned dataset (clean_data) is created from data.copy().
Remove negative values in numeric columns (keeps only >= 0)
Remove impossible values: BMI must be > 0
Life expectancy must be between 20 and 120
Schooling must be between 0 and 25
GDP must be > 0

**Feature Engineering**
A prepared dataset (data_prep) is created from clean_data.copy().
Engineered features:

**Thinness_avg** was created by taking the average of Thinness_ten_nineteen_years and Thinness_five_nine_years
Thinness_avg = (Thinness_ten_nineteen_years + Thinness_five_nine_years) / 2
Since they are highly correlated with one another and it reduces redundancy and creates a single malnutrition indicator.

**Vaccination index** was created by taking the average of Polio, Hepatitis_B, Diphtheria values
Vaccination_index = (Polio + Hepatitis_B + Diphtheria) / 3
It was done because it summarises vaccination coverage, it is easier to interpret as a health system strength proxy.

**Log population**. The log og the population was taken to reduce the skewness.
Log_population = log1p(Population_mln)
Population is highly skewed, log transform reduces skewness and helps some models.

**Total child deaths** was created by combining the Infant_deaths and Under_five_deaths values as one
Total_child_deaths = Infant_deaths + Under_five_deaths
The reason is that it captures overall child mortality burden.

**Then the original variables replaced by engineered features are dropped (e.g., thinness components, vaccination components, infant/under-five deaths, population).**

**Encoding Categorical Variables (One-Hot Encoding)**
Country and Region are converted using:
pd.get_dummies(..., drop_first=True, dtype=int)

Why one-hot encoding:
Country and Region are nominal (no inherent order).
Label encoding would introduce artificial ranking (e.g., Country 1 < Country 2), which is incorrect.
Why drop_first=True:
Drops one category per variable to avoid the dummy variable trap (perfect multicollinearity in linear models).

Why dtype=int:
Converts dummy columns to numeric 0/1, ensuring compatibility across all models and metrics.

**Defining X and y, and Train/Test Split**
y = Life_expectancy
X = all other columns
Split: 80% train / 20% test with random_state=42
Why split: The test set simulates unseen data and provides a more honest estimate of generalisation.

**Scaling**
The notebook includes manual scaling using StandardScaler to create:
X_train_scaled, X_test_scaled

**Important note**: For model evaluation using cross-validation, scaling is best done inside pipelines (to prevent leakage). The modelling section uses pipelines for this reason.

**Model Development and Cross-Validation (K-Fold)**
Models implemented:
Linear Regression (pipeline with scaling)
SVR (pipeline with scaling)
Random Forest
Decision Tree
XGBoost

Cross-validation uses:
KFold(n_splits=5, shuffle=True, random_state=42)
cross_validate(...) with scoring for MAE, MSE (converted to RMSE), R², and MAPE

**Why k-fold CV**
Reduces dependence on a single train/test split
Provides more reliable performance estimates (average over folds)
Required by the coursework brief

**Results Summary and Visualisation**
Outputs produced:
Model comparison table: RMSE, MAE, MAPE, R²
Combined bar + line chart (errors as bars, R² as line)

Multiple metrics were used because:
RMSE penalises large errors
MAE is robust and interpretable
MAPE provides relative % error
R² shows variance explained

**Feature Importance (XGBoost)**
Feature selection for XGBoost uses built-in model importance.
It trains XGBoost (after optionally dropping country/region dummies)
Extracts feature_importances_
Displays top 20 features and plots them

**Why this method**
XGBoost provides an embedded, model-based feature importance measure.
It is suitable for non-linear tree ensembles and interpretable for presentation/reporting.

Why drop Country/Region for XGBoost importance:
Country and region dummies can dominate importance due to strong proxy effects.
Dropping them helps focus interpretation on socioeconomic/health drivers (aligned with project objective).

**Hyperparameter Tuning (RandomizedSearchCV)**
RandomizedSearchCV is used with:
cv=5 (5-fold cross-validation inside tuning)
scoring based on RMSE (neg_root_mean_squared_error)
parameter distributions for RF, SVR, DT, XGBoost

Why RandomizedSearchCV:
More efficient than GridSearch for large search spaces (especially XGBoost)
Achieves strong tuning results with fewer evaluations
Final tuned models are evaluated on the test set and plotted.

**Summary**
Mean imputation used to preserve data size and keep workflow simple and explainable.
Rule-based cleaning used to remove impossible values rather than aggressively dropping statistical outliers.
Feature engineering + dropping originals used to reduce redundancy and multicollinearity.
One-hot encoding used because Country/Region have no order; drop_first=True avoids dummy trap.
Pipelines used for LR and SVR to prevent scaling leakage during CV.
5-fold CV used to provide robust performance estimates (required by brief).
RandomizedSearchCV used for hyperparameter tuning efficiency.
XGBoost feature importance used as an embedded method for interpreting drivers.

**Notes on Reproducibility**
Random seeds (random_state=42) were used for consistent results.
Ensure the dataset file name and path match the read_csv(...) call.
If running in Colab, upload the CSV to the session or mount Google Drive.

**Limitations and Areas for Improvement**
Mean imputation may bias results if missingness is systematic.
Extremely high R² values may indicate strong feature redundancy or country proxy effects.
Country/Region can inflate accuracy; results should be interpreted with care.

**Future improvements**
Evaluate generalisation by leaving out entire countries (grouped CV)
Use SHAP values for deeper model explainability (especially XGBoost)
