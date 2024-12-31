# Supervised-Learning-Classification-using-Africa-Systemic-BankingCrisis-Dataset
### Overview
The purpose of this project is to develop a machine learning model that predicts the likelihood of a systemic crisis emergence in African countries based on various economic indicators. The scope of this project is limited to analyzing the provided dataset, which focuses on banking, debt, financial, inflation, and systemic crises in 13 African countries from 1860 to 2014. The problem addressed by this project is the need to identify early warning signs of systemic crises in African countries, which can help policymakers and stakeholders take proactive measures to prevent or mitigate the impact of such crises.

### Key Features
- Data Exploration: The exercise involves performing basic data exploration, including handling missing and corrupted values, removing duplicates, handling outliers, and encoding categorical features.
- Dataset Analysis: The exercise involves analyzing a dataset related to systemic crises in African countries.
- Feature Selection: The exercise involves selecting relevant features from the dataset to use in the machine learning model.
- Classification Problem: The exercise involves solving a classification problem, where the target variable is a binary outcome (e.g., crisis or no crisis).
- Machine Learning Model Development: The goal is to develop a machine learning model that predicts the likelihood of a systemic crisis emergence.
- Model Evaluation: The performance of the machine learning model will be evaluated using relevant metrics.
- Model Improvement: The exercise encourages discussing alternative ways to improve the model's performance.

### Dataset
Details about the dataset:
- Source: [https://drive.google.com/file/d/1fTQ9R29kgAhInFO0HMqvkcAfSZWg6fCx/view]
- Description: The structure of the dataset is such that it has 1059 entries, 14 Columns, a RangeIndex with Data Types of Float64 in 3 columns (exch_usd, gdp_weighted_default, inflation_annual_cpi), Int64 in 8 columns (country_number, year, systemic_crisis, domestic_debt_in_default, sovereign_external_debt_default, independence, currency_crises, inflation_crises), and Object in 3 columns (country_code, country, banking_crisis)

### Requirements
A list of required libraries and dependencies:
Required Libraries
1. pandas
2. numpy
3. matplotlib
4. seaborn
5. scikit-learn
6. ydata-profiling
Machine Learning Classification Models Used
1. Logistic Regression
2. Decision Trees
3. Random Forest
4. Support Vector Machines (SVM)
5. K-Nearest Neighbors (KNN)
 
### How to Run
1. Clone this repository:  
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:  
   ```bash
   jupyter notebook Africa Systemic Crisis Prediction.ipynb
   ```

### Results
#### Key Findings
Machine Learning Model Performance
- Five machine-learning classification models were trained and evaluated: Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
- The models were assessed using accuracy, precision, recall, and F1-score metrics.

Best-Performing Model
- The Decision Tree Classifier achieved the highest accuracy, with an average accuracy of 0.9434.
- The model demonstrated good performance in terms of precision (0.98 for class 0 and 0.61 for class 1), recall (0.95 for class 0 and 0.82 for class 1), and F1-score (0.97 for class 0 and 0.70 for class 1).
- The confusion matrix showed 186 true positives, 9 false positives, 3 false negatives, and 14 true negatives.

Feature Importance
1. systemic_crisis: Highly correlated with banking_crisis, indicating a strong relationship between systemic crises and banking crises.
2. domestic_debt_in_default: Moderately correlated with banking_crisis, suggesting a relationship between domestic debt defaults and banking crises.
3. exch_usd: Moderately correlated with banking_crisis, indicating a connection between exchange rates and banking crises.
4. sovereign_external_debt_default: Moderately correlated with systemic_crisis, which is highly correlated with banking_crisis, suggesting a relationship between sovereign external debt defaults and banking crises.

Limitations and Future Work
- The dataset was limited to 13 African countries, and future work could involve expanding the dataset to include more countries.
- Additional features, such as macroeconomic indicators or political stability metrics, could be incorporated to improve model performance.
