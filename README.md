# 🏡 House Prices - Advanced Regression Techniques

A comprehensive machine learning solution for the Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal is to predict the final sale price of homes in Ames, Iowa, based on 79 explanatory variables encompassing almost every aspect of residential properties.

**Kaggle Notebook:** [hngvvan / house-prices-advanced-regression-techniques](https://www.kaggle.com/code/hngvvan/house-prices-advanced-regression-techniques)

## 📊 Results & Evaluation

| Model / Submission | Public Score (RMSLE) |
|---|---|
| **Kaggle Notebook Version** | **0.15004** |
| `submission.csv` (Direct Upload) | 0.15013 |

- **Evaluation Metric:** Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. A score of `~0.15` indicates that the model's predictions are off by approximately 15% on average across the test set.
<img width="1420" height="704" alt="score" src="https://github.com/user-attachments/assets/3020e15c-7c26-41be-a9a3-a8e64167513e" />

---

## 🛠 Approach & Methodology

### 1. Exploratory Data Analysis (EDA)
Thorough data exploration was conducted to understand the complex relationships within the dataset's 79 features before modeling:
- **Target Variable Analysis:** The target variable `SalePrice` was analyzed and found to be heavily right-skewed, which is typical for real estate pricing. This insight directly motivated the target transformation step later in the pipeline.
- **Feature Selection via Correlation:** A detailed correlation analysis was performed to isolate the most influential numerical features driving the house prices (such as `OverallQual`, `GrLivArea`, `GarageCars`, and `TotalBsmtSF`). Conversely, features with negligible correlation to the target were aggressively dropped to reduce multidimensional noise and improve model generalization.

### 2. Data Preprocessing
To ensure a robust, reproducible workflow and absolutely prevent data leakage, a scikit-learn `ColumnTransformer` was utilized to route different data types through specific preprocessing steps:
- **Numerical Features:** Addressed missing values using median imputation (which is robust to outliers) and standardized the scale of the data using `StandardScaler`.
- **Ordinal Categorical Features:** Missing values were filled with a constant string ("None" for features like pool or garage quality where missing implies absence). An `OrdinalEncoder` was then applied to preserve and mathematically represent the natural ranking of house qualities (e.g., `None < Po < Fa < TA < Gd < Ex`).
- **Nominal Categorical Features:** Filled missing values with a constant and applied `OneHotEncoder`. The encoder was configured to ignore unknown categories, ensuring the model handles unseen variations in the hidden test set smoothly without breaking.

### 3. Modeling
The core predictive engine is a `RandomForestRegressor`. This tree-based ensemble method was chosen for its inherent ability to capture non-linear relationships between features and its robustness against overfitting. To maintain a clean architecture, the entire workflow—from missing value imputation to the final prediction—was encapsulated within a single scikit-learn `Pipeline`.

### 4. Target Transformation
Given that the competition metric is RMSLE (Root Mean Squared Logarithmic Error), the model was trained directly on the natural logarithm of the target variable using `np.log1p(y_train)`. This transformation not only aligns the training objective with the evaluation metric but also normalizes the right-skewed distribution of `SalePrice`. After generating predictions, the values were converted back to standard dollar amounts using the exponential function `np.expm1(val_pred)`.

---

## 📁 Project Structure

```text
├── House Prices - Advanced Regression Techniques.ipynb   # Main Jupyter Notebook containing the full EDA, training, and prediction workflow
├── README.md                                             # Project documentation and summary
└── submission.csv                                        # Final generated predictions for Kaggle scoring
```

---

## 🚀 How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies (ensure you have Python 3.8+ installed):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. Launch Jupyter Notebook and run the cells sequentially:
   ```bash
   jupyter notebook "House Prices - Advanced Regression Techniques.ipynb"
   ```
