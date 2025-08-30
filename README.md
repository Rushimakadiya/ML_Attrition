# Employee Attrition Prediction using Random Forest

This notebook implements a machine learning pipeline to predict employee attrition using a Random Forest classifier. The workflow covers data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation with detailed metrics and visualizations.

## 📦 Workflow Overview

- **Data Loading:** Reads `Attrition_dataset.csv` and cleans unwanted columns.
- **Resampling:** Balances the dataset by upsampling the minority class (`Attrition = Yes`).
- **Encoding:** Applies label encoding and one-hot encoding to categorical features.
- **Outlier Handling:** Replaces outliers in numeric columns with mean values.
- **Feature Selection:** Uses Chi-Squared and ANOVA F-value to select top features.
- **Scaling:** Scales features using MinMaxScaler.
- **Model Training:** Splits data, tunes hyperparameters with GridSearchCV, and trains a Random Forest classifier.
- **Evaluation:** Outputs accuracy, precision, F1 score, and confusion matrix. Plots a heatmap for confusion matrix and prints a detailed classification report.

## 🧪 Results

- **Best Model:** Hyperparameters are optimized using GridSearchCV for maximum F1 score.
- **Metrics:**
  - **Accuracy:** Measures overall correctness of predictions.
  - **Precision:** Measures correctness for positive predictions (`Yes Attrition`).
  - **F1 Score:** Harmonic mean of precision and recall.
  - **Confusion Matrix:** Visualizes true/false positives and negatives.
  - **Classification Report:** Shows precision, recall, F1-score for each class (`No Attrition`, `Yes Attrition`).

Example output:
```
Classification Report:
                 precision    recall  f1-score   support

No Attrition         0.98      0.96      0.97       370
Yes Attrition        0.96      0.98      0.97       370

      accuracy                           0.97       740
     macro avg       0.97      0.97      0.97       740
  weighted avg       0.97      0.97      0.97       740
```
[Confusion Matrix Heatmap](confusion_matrix.png)

## 🚀 How to Run

1. Place `Attrition_dataset.csv` in the same directory.
2. Open and run all cells in [Random_forest.ipynb](Attrition/Random_forest.ipynb) using Jupyter Notebook.
3. Results and plots will be displayed inline.

## 🛠️ Dependencies

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

Install with:
```
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📄 License

MIT License

---

*Predict employee attrition with interpretable metrics and visualizations using Random Forest.*
