# -*- coding: utf-8 -*-

# ☀️ Solar Power Generation — EDA File

""" **Objective :**

The variable to be predicted is continuous (energy production). Therefore, this is an Regression project. The primary goal is to model energy production as a function of environmental variables.

The goal of this project is to predict continuous solar energy output (power_generated) in Joules per 3 hours using environmental variables such as present in the dataset of "solar Power Generation"

**Dataset Discription :**

The file solarpowergeneration.csv contains the data for this example. Here, the number of variables (columns) is 10, and the number of instances (rows) is 2920. We have the following variables for this analysis:

•	distance_to_solar_noon : in radians.

•	temperature : daily average temperature, in degrees Celsius.

•	wind_direction : daily average wind direction, in degrees (0-360).

•	wind_speed : daily average wind speed, in meters per second.

•	sky_cover : on a five-step scale from 0 to 4, with 0 completely clear and 4 wholly covered.

•	visibility : in kilometers.

•	humidity : in percentage.

•	average_wind_speed : average wind speed during the 3 hours de measure was taken in meters per second.

•	average_pressure : average barometric pressure during the 3 hours the measure was taken in, in mercury inches.

•	power_generated(Target) : in Jules for each 3 hours
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# load CSV
df = pd.read_csv("solarpowergeneration.csv")
df.head()

"""# Data Overview"""

print("\nDataSet Dimensions:")
print(df.shape)

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing values per column:\n", df.isnull().sum())

print("\nDuplicate Rows:\n",df.duplicated().sum())

"""Notes:

* The dataset has 2920 rows and 10 columns.

* Only 1 missing value (in average_wind_speed_period) → filled with mean.

* Column names standardized (underscores only).

* No duplicates detected.

* All features are numeric and continuous, suitable for regression.

# Data Cleaning
"""

# Filling the single missing value of average-wind-speed-(period) with column mean
df['average-wind-speed-(period)'].fillna(df['average-wind-speed-(period)'].mean(), inplace=True)

print("\nMissing values after applying mean:\n", df.isnull().sum())

# Cleaning the column names (replace special characters "-" with "_" underscores) to have cleaned column names
df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_', regex=True)

# Confirm
print("\nCleaned Columns:\n", df.columns)

"""Notes:

* Clean naming ensures smooth coding later (no spaces or parentheses).

* Filled missing values using mean imputation to retain dataset size.

* Dataset now clean and ready for visualization.

# Data Visualization
"""

# Univariate analysis with histogram plots
# the examination of a single variable (column) in the dataset to understand its characteristics
num_cols = df.select_dtypes(include='number').columns

df[num_cols].hist(
    bins=30,
    figsize=(14, 10),
    color='skyblue',
    edgecolor='black',
    layout=(len(num_cols)//3 + 1, 3)  # optional: control subplot grid
)

plt.suptitle("Distribution of Numeric Variables", fontsize=16, y=1.02)
plt.tight_layout(pad=2.0)
plt.show()

"""Insights:

* temperature is roughly normal (centered near 25°C).

* humidity is slightly right-skewed — high humidity can reduce irradiance.

* visibility (km) is slightly right-skewed mostly clear visibility.

* sky_cover shows a discrete 0–4 scale distribution(partly cloudy).

* distance_to_solar_noon is bimodal which is expected daily solar cycle.

* power_generated (j) varies widely — consistent with day/night cycles and energy peaked at midday.

* wind_speed,average_wind_speed_period_,average_pressure_period are roughly normal and stable daily readings.

NOTE : For Bivariate analysis the target column (power_generated) is compared with all the other columns or variables of the dataset (solarpowergeneration)
"""

# bivariate analysis with scatter plots
# the examination of the relationship between two variables in a dataset.
features = ['distance_to_solar_noon','temperature','wind_direction','wind_speed','sky_cover',
            'visibility','humidity','average_wind_speed_period_','average_pressure_period_']

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.scatterplot(x=col, y='power_generated', data=df, ax=axes[i], color='teal')
    axes[i].set_title(f"{col.replace('_',' ').title()} vs Power Generated")
    axes[i].set_xlabel(f"{col.replace('_',' ').title()} ({'°C' if col=='temperature' else '%' if col=='humidity' else 'radians' if col=='distance_to_solar_noon' else 'km' if col=='visibility' else 'm/s' if 'wind' in col else 'scale'})")
    axes[i].set_ylabel("Power Generated (Joules per 3 hours)")

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
plt.show()

"""**Insights**

[high =  ↑ , low =  ↓ ]

* Distance to solar noon ↓ → Power ↑ (sun at zenith = max output)
                            strong negative trend — energy peaks near noon.

* Sky cover ↑ → Power ↓ (clouds block irradiance)clear skies → high output; overcast → steep drop in energy.

* Temperature ↑ → Power ↑ (clearer skies)positive relation (clearer skies = higher irradiance).

* Visibility ↑ → Power ↑ (less haze) better visibility = higher power generation.

* Humidity ↑ → Power ↓ (scattering)higher humidity lowers energy output due to air moisture.

* Wind speed and average wind speed period shows weak positive correlation with power generation.Weak positive trend (minor cooling effect) small influence.

* wind direction and average pressure period shows no visible pattern.a slightly low correlation among the others.



"""

# Multivariate analysis: pair plot for numeric features including target variable
sns.pairplot(df, vars=['temperature', 'wind_speed', 'sky_cover', 'humidity', 'power_generated','average_wind_speed_period_','average_pressure_period_','visibility','wind_direction','distance_to_solar_noon'], hue='sky_cover', palette='viridis')
plt.suptitle('Pairplot: Feature Interactions with Target', y=1.02)
plt.show()

"""**Insights**

* Color-coded by sky_cover → helps identify cloudy vs clear-day patterns.

* Strong clustering near low distance_to_solar_noon and high power_generated when sky_cover is low.

* Clear separations show sky_cover is a dominant feature.

* temperature and humidity inversely interact, as expected.
"""

# box plots for outlier detection for all numeric columns
n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2 * n_rows))
axes = axes.flatten()

colors = sns.color_palette("Set2", len(num_cols))

# Plot boxplots
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i], color=colors[i])
    axes[i].set_title(f'Box Plot for {col}')

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
plt.show()

"""**Outlier Analysis :**

* temperature	>> few mild outliers → very hot days
* humidity	>> few high values (>90%)
* visibility	>> some low outliers (foggy days)
* power_generated >>	upper outliers = peak sunlight intervals
* distance_to_solar_noon >>	no true outliers — cyclic variable
* wind_speed	>> stable, no outliers

NOTE : Outliers appears natural (extreme values represent natural environmental variance), not data errors so no removal required.
"""

# Correlation heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='crest', fmt='.2f')
plt.title("Correlation Heatmap — Environmental Features vs Power Generated")
plt.show()

"""**Interpretation :**

Feature---------| | |-------Corr with Power Generated--------| | |----Importance

---




distance_to_solar_noon------	[−0.88]-----------------------🔥 Very Strong (negative)

sky_cover	-------------------[−0.65]-----------------------🔥 Strong (negative)

temperature	----------------[+0.55]-----------------------🔥 Strong (positive)

visibility	---------------------[+0.35]-----------------------	✅ Moderate

humidity	---------------------[−0.32]-----------------------	✅ Moderate

wind_speed	------------------[+0.18]-----------------------	⚪ Weak

average_wind_speed_period	--[+0.10]-----------------------	⚪ Weak/Duplicate

average_pressure_period	-----[+0.05]-----------------------	🚫 Negligible

wind_direction	----------------[+0.02]-----------------------	🚫 Negligible
"""

# Feature Importance (Correlation-Based Ranking)
corrs = df.corr()['power_generated'].drop('power_generated').sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=corrs.values, y=corrs.index, palette='viridis')
plt.title("Feature Correlation with Power Generated")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.show()

"""**Most Important Features**

1. distance_to_solar_noon → highest impact; directly tied to sun’s position.

2. sky_cover → inversely proportional to irradiance.

3. temperature → indirect proxy for clear sky and solar intensity.

4. visibility → helps capture atmospheric clarity.

5. humidity → inversely affects energy generation (scattering light).

**Less Useful / Redundant Features**

1. wind_direction → adds random noise (no logical link to power).

2. average_wind_speed_period → overlaps with wind_speed; may cause collinearity.

3. average_pressure_period → very weak correlation with sunlight intensity.

**Interpretation:**

Highest positive impact → temperature, visibility

Highest negative impact → distance_to_solar_noon, sky_cover

Negligible impact → wind_direction, average_pressure_period

# Feature Selection
"""

final_list_columns = df[['distance_to_solar_noon', 'temperature', 'sky_cover', 'visibility', 'humidity', 'wind_speed','power_generated']]

print("final_list_columns shape:",final_list_columns.shape)

# data selection or data partition
selected_features = ['distance_to_solar_noon','temperature','sky_cover','visibility','humidity','wind_speed']
X = df[selected_features]
y = df['power_generated']

# splitting the data into train and test splits
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

"""# Standardization"""

# Scalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Assumptions and Feature Engineering Notes

- Dataset cleaned, analyzed, and correlations established.
- Irrelevant features (wind_direction, average_pressure_period, average_wind_speed_period) identified and dropped.
- Missing values imputed with mean to avoid bias.
- Outliers were inspected via boxplots; no extreme values were removed as they appear valid.
- Dropped variables shown to have low correlation with the target or high multi-collinearity based on the heatmap.
- Key influencing features retained (distance_to_solar_noon, temperature, sky_cover, visibility, humidity, wind_speed).
- StandardScaler chosen to normalize features due to varied magnitudes and to improve model convergence. All numerical features standardized (mean=0, std=1).


# EDA Summary Report

- Data is clean with minimal missing values handled.
- Univariate and bivariate analyses reveal feature distributions and key predictors for power generation.
- Multivariate analysis shows interactions and correlations, guiding feature selection.
- Feature engineering decisions are based on data understanding and correlation analysis.
- Strongest predictors: distance_to_solar_noon (−), sky_cover (−), temperature (+).
- Moderate: visibility (+), humidity (−).
- Weak or irrelevant: wind_direction, pressure, avg_wind_speed_period.
- Data is now prepared and scaled for modeling.
- scaled_df can now be split into train-test for model building.

- Suggested models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost.

- Evaluate using MAE, RMSE, R² metrics.

# PHASE 2 : Model Building and Evaluation
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01)
    }

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append([name, mae, rmse, r2])
    print(f" {name} Results:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\n{'-'*40}")

"""**Insights**

- Started with simple baseline models (Linear, Ridge, Lasso).

- These help us understand a “minimum” performance expectation before advanced models.

- Ridge and Lasso add regularization — useful for reducing overfitting.

- The R² ≈ 0.61 indicates these models explain around 61% of energy variation — acceptable but room for improvement.
"""

advanced_models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}
for name, model in advanced_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append([name, mae, rmse, r2])
    print(f" {name} Results:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\n{'-'*40}")

"""**Insights**

- Advanced ensemble models capture non-linear patterns effectively.

- Random Forest, XGBoost, and LightGBM outperform baseline models significantly.

- XGBoost achieved R² ≈ 0.89, showing that boosting improves learning accuracy.

- LightGBM gave nearly similar performance with faster training speed.
"""

# Compile results
all_results = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²"])
print("\n Model Comparison:")
print(all_results.sort_values(by="R²", ascending=False))

plt.figure(figsize=(8,5))
sns.barplot(data=all_results.sort_values("R²", ascending=False), x="R²", y="Model", palette="viridis")
plt.title("Model Performance Comparison (R² Score)")
plt.show()

# hyperparameter tuning
params = {
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

xgb = XGBRegressor(random_state=42)
search = RandomizedSearchCV(xgb, params, cv=5, scoring='r2', n_iter=20, n_jobs=-1, verbose=1)
search.fit(X_train_scaled, y_train)

best_xgb = search.best_estimator_
print("\n Best Parameters:", search.best_params_)
print("Best R² on Test Set:", r2_score(y_test, best_xgb.predict(X_test_scaled)))

"""**Insights**

- RandomizedSearchCV performs random hyperparameter combinations and uses 5-fold CV for fair testing.

- Best parameters are chosen based on highest R².

- After tuning, XGBoost’s R² increased to around 0.90, meaning the model now explains 90% of total variation in power generation.

- Cross-validation ensures the model is stable and not just performing well on one train-test split.
"""

# cross validation
scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCross-validated R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

"""**Insights**

- Cross-validation R² ≈ 0.90 shows consistent performance across different folds.

- Confirms the model generalizes well — predictions are stable on unseen data.

# Model Predictions
"""

# prediction
sample_data = pd.DataFrame({
    'distance_to_solar_noon': [0.065],
    'temperature': [69],
    'sky_cover': [0],
    'visibility': [10.0],
    'humidity': [21],
    'wind_speed': [7.5]
})

# Apply scaling
sample_scaled = scaler.transform(sample_data)

predicted_power = best_xgb.predict(sample_scaled)
print(f"\n🔆 Predicted Solar Power Generated: {predicted_power[0]:.2f} Joules (per 3-hour period)")

"""**Insights**

- Model predicts continuous energy output in Joules, making it realistic for engineering analysis.

- Example: Predicted = 30083.12 Joules vs Actual = 30069 J → only 1.3% difference.

- Shows that the model generalizes well even on unseen environmental conditions.
"""

# inputs for the manual predictions
df.head(10)

# Predicted vs Actual Plot
y_pred = best_xgb.predict(X_test_scaled)

# Calculate residuals
residuals = y_test - y_pred

# comparison
comparison_df = pd.DataFrame({
    'Actual (Joules)': y_test,
    'Predicted (Joules)': y_pred
}).reset_index(drop=True)

# Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='Actual (Joules)', y='Predicted (Joules)', data=comparison_df, color='teal', alpha=0.7, s=60)

# Diagonal reference line (Perfect prediction)
plt.plot([comparison_df.min().min(), comparison_df.max().max()],
         [comparison_df.min().min(), comparison_df.max().max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title("Actual vs Predicted Solar Power Generation (XGBoost)", fontsize=13)
plt.xlabel("Actual Power Generated (Joules)")
plt.ylabel("Predicted Power Generated (Joules)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# some sample of comparisons actual vs predicted data variance
print("🔹 Sample Comparison:")
print(comparison_df.head(10))

"""**Insights**

- Most points align near the red diagonal, proving high accuracy.

- Slight underpredictions at very high energy outputs are normal for real solar data.

- The visualization confirms the tuned XGBoost model fits the pattern very well.
"""

plt.subplot(1,1,1)
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("Residual Distribution")
plt.xlabel("Residuals (Joules)")
plt.tight_layout()
plt.show()

# feature importance
importances = best_xgb.feature_importances_
sns.barplot(x=importances, y=selected_features, palette='viridis')
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

"""**Insights**

- distance_to_solar_noon and sky_cover dominate the model — most influential on energy output.

- Confirms physics-based logic: clearer skies and higher sun angle = more solar energy.

- Humidity and temperature also show moderate influence

# Final Summary
- Multiple regression techniques were tested to predict solar energy generation.  
- Ensemble methods (XGBoost & LightGBM) clearly outperformed simple linear models.  
- After applying **cross-validation and hyperparameter tuning**, the **XGBoost model** achieved:
  - MAE ≈ 1600  
  - RMSE ≈ 3300  
  - R² ≈ 0.90  
- The model effectively predicts continuous solar energy output with minimal error(~ 1.3% of Mean Square Error).  
- The model is now ready for deployment.
"""


import joblib

# Save model and scaler
joblib.dump(best_xgb, 'model.pkl')       # or use your best model variable
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and Scaler saved successfully!")
