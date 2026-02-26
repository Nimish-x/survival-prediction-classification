# Maritime Survival Prediction Classification

## 📌 Problem Statement
The goal of this project is to build a classification model capable of predicting the survival `Outcome` of passengers in a maritime disaster scenario based on various passenger attributes such as demographics, ticket class, and cabin location.

## 🚀 Solution Approach
The project follows a structured machine learning pipeline:

### 1. Exploratory Data Analysis (EDA)
- Inspected the dataset and its features to understand the datatypes and missing value prevalence.
- Analyzed the distributions and target variables.

### 2. Feature Engineering & Data Preprocessing
- **Handling Missing Values:** Missing numerical values in the `Age` column were imputed using the median strategy via `SimpleImputer` to provide robustness against outliers.
- **Feature Extraction & Creation:**
  - `Deck`: Extracted the deck level (first letter) from the `Berth` attribute.
  - `AgeGroup`: Segmented continuous age data into categorical bins ('Child', 'Teen', 'Adult', 'MiddleAged', 'Senior').
  - `TicketGroupSize`: Mapped the `CLass` to frequency counts to capture group sizes.
- **Encoding & Scaling:**
  - Applied One-Hot Encoding (`pd.get_dummies()`) to all categorical features (Gender, Class, Berth, Boarding Port, Title, AgeGroup, Deck).
  - Scaled numerical features using `StandardScaler` to ensure standardized inputs.

### 3. Model Building and Evaluation
Three distinct algorithms were evaluated:
- **Random Forest Classifier**: A robust ensemble method. Evaluated via 8-fold cross-validation and Stratified K-Fold.
- **Logistic Regression**: Hyperparameter tuned using `RandomizedSearchCV` across a space of `C` and `l1_ratio` (ElasticNet penalty).
- **LightGBM Classifier**: A fast, distributed, high-performance gradient boosting framework.

The final best-performing model was used to generate predictions on the unseen test dataset.

## 🛠️ Tools & Technologies Used
- **Language**: Python 3
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning library**: `scikit-learn` (Model selection, preprocessing, metrics, RandomForest, LogisticRegression)
- **Gradient Boosting**: `lightgbm`

## 📂 Project Structure
- `classification_notebook.ipynb`: The main structured Jupyter Notebook containing all the analysis, feature engineering, and model training.
- `submission.csv`: (Generated after running the notebook) Contains the final predictions on the test dataset.

## ⚙️ How to Run
1. Ensure you have the datasets downloaded as CSV files (`maritime_train.csv` and `maritime_test.csv`) placed in the correct directory.
2. Install necessary dependencies from the Tools list.
3. Run the notebook cells sequentially to train the models and generate the prediction CSV.
