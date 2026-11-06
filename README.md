# 🏡 House Prices – Advanced Regression Techniques

## 📌 Project Overview

This project focuses on developing a Machine Learning regression model to predict residential house prices using the [Kaggle House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset.

The objective is to predict the final sale price (`SalePrice`) of residential properties in Ames, Iowa, based on 79 explanatory variables describing various aspects of each property, including building quality, living area, location, and facilities.

This project covers the end-to-end Machine Learning workflow, including Exploratory Data Analysis (EDA), feature selection, data preprocessing, categorical encoding, target transformation, model training, and prediction generation.

**Kaggle Notebook:** [House Prices – Advanced Regression Techniques](https://www.kaggle.com/code/hngvvan/house-prices-advanced-regression-techniques)

---

## 📊 Results & Evaluation

| Model / Submission | Public Score (RMSLE) |
| :--- | :---: |
| Kaggle Notebook | **0.12953** |

**Evaluation Metric: Root Mean Squared Logarithmic Error (RMSLE)**

The competition evaluates predictions using RMSLE, which measures the difference between the logarithmic values of predicted and actual house prices.

RMSLE penalizes relative prediction errors and reduces the influence of extremely expensive properties compared with metrics calculated directly on the original price scale.

A lower RMSLE indicates better predictive performance.

- **Public Score:** 0.12953
- **Model:** GradientBoostingRegressor
- **Prediction Target:** `SalePrice`

---

## 🛠️ Methodology & Implementation

### 1. Exploratory Data Analysis (EDA)

Conducted exploratory data analysis to understand the dataset's structure, feature distributions, missing values, and relationships with the target variable.

Key analysis areas:

- **Target Variable Distribution:** Analyzed the distribution of `SalePrice` and identified significant right skewness.
- **Correlation Analysis:** Examined relationships between numerical features and house prices.
- **Feature Importance Exploration:** Investigated influential variables such as `OverallQual`, `GrLivArea`, `GarageCars`, and `TotalBsmtSF`.
- **Missing Value Analysis:** Identified features containing missing values and examined their potential meanings.
- **Feature Selection:** Used correlation analysis to identify relevant numerical features and reduce unnecessary dimensionality.

### 2. Data Preprocessing

Implemented a preprocessing architecture using Scikit-learn's `ColumnTransformer` to apply different transformations to numerical, ordinal categorical, and nominal categorical features.

**Numerical Features:**
- Applied `SimpleImputer(strategy='median')` to handle missing values.
- Used `StandardScaler` to standardize numerical feature scales.

**Ordinal Categorical Features:**
- Applied missing value imputation using a constant value such as `"None"` where missing values represent the absence of a property feature.
- Used `OrdinalEncoder` to transform categorical variables into numerical representations while preserving their natural order.
- Maintained the ranking of property quality levels, for example: `None < Po < Fa < TA < Gd < Ex`.

**Nominal Categorical Features:**
- Applied constant-value imputation to handle missing values.
- Used `OneHotEncoder(handle_unknown='ignore')` to convert categorical variables into numerical representations.
- Configured the encoder to handle previously unseen categories in the test dataset.

**Pipeline Architecture:**

The preprocessing transformations were integrated into a Scikit-learn `ColumnTransformer` and `Pipeline` to maintain a consistent workflow during training and prediction.

This approach also helps prevent data leakage by fitting preprocessing transformations on the training data rather than the entire dataset before validation.

### 3. Model Development & Training

**Selected Model: GradientBoostingRegressor**

Used `GradientBoostingRegressor` as the core regression algorithm.

Gradient Boosting is an ensemble learning technique that builds decision trees sequentially, with each subsequent tree attempting to correct the errors made by the previous ensemble.

Key advantages:

- Captures non-linear relationships between property features and sale prices.
- Models complex interactions between numerical and categorical variables.
- Combines multiple weak learners to improve predictive performance.
- Works effectively with structured tabular datasets.

The complete preprocessing and modeling workflow was encapsulated within a single Scikit-learn `Pipeline` to ensure consistency and reproducibility.

### 4. Target Transformation

The original `SalePrice` distribution is heavily right-skewed, while the competition evaluates predictions using RMSLE.

To address this, the target variable was transformed using the natural logarithm:

```python
y_train = np.log1p(y_train)
```

This transformation:

- Reduces the right skewness of the target distribution.
- Reduces the influence of large price values.
- Aligns the training objective more closely with the logarithmic evaluation metric.

After model training, predictions were converted back to the original price scale using the inverse transformation:

```python
val_pred = np.expm1(val_pred)
```

This ensures that the final predictions represent actual house prices rather than logarithmic values.

### 5. Final Prediction & Submission

After completing model development, the trained pipeline was used to generate predictions for the competition's test dataset.

The predicted values were transformed back to the original price scale and exported to `submission.csv` using the required Kaggle submission format.

---

## 💻 Tech Stack

| Category | Technologies |
| :--- | :--- |
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook |
| Platform | Kaggle |

---

## 📁 Project Structure

```text
House-Prices-Advanced-Regression-Techniques/
│
├── House Prices - Advanced Regression Techniques.ipynb
│                                      # Main notebook: EDA, feature selection,
│                                      # preprocessing, model training and prediction
│
├── submission.csv                    # Generated Kaggle submission file
│
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <https://github.com/hngvvan/House-Prices-Advanced-Regression-Techniques.git>
cd House-Prices-Advanced-Regression-Techniques
```

### 2. Install Dependencies

Ensure that Python 3.8 or later is installed.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Download the Dataset

Download the competition dataset from Kaggle:

[House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Place the downloaded CSV files in the directory expected by the notebook.

### 4. Run the Notebook

```bash
jupyter notebook "House Prices - Advanced Regression Techniques.ipynb"
```

Run the notebook cells sequentially to reproduce the data analysis, preprocessing, model training, and prediction workflow.

---

## 🎯 Key Learnings

Through this project, I gained practical experience in:

- Performing Exploratory Data Analysis on structured regression datasets.
- Analyzing skewed target distributions and numerical feature correlations.
- Selecting relevant features based on correlation analysis.
- Handling missing values across different feature types.
- Applying ordinal and one-hot encoding to categorical variables.
- Building preprocessing pipelines using `ColumnTransformer` and `Pipeline`.
- Applying logarithmic target transformation for regression tasks.
- Training and evaluating Gradient Boosting regression models.
- Generating predictions and preparing submissions for Kaggle competitions.

---
