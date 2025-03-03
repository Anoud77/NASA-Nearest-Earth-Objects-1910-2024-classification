# NASA-Nearest-Earth-Objects-1910-2024-classification

## Project Overview
This project involves comparing four different machine learning models to classify a dataset of Earth objects (e.g., asteroids, comets) based on various features. The goal is to evaluate and select the best-performing model by comparing key metrics such as accuracy, precision, recall, and F1 score. Additionally, confusion matrices are visualized to give insights into the models' performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Approach](#approach)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Comparison](#model-comparison)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Comparison](#results-and-comparison)
- [Insights and Key Findings](#insights-and-key-findings)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [Acknowledgments](#acknowledgments)

## Dataset Description
The dataset used in this project contains information about Earth objects and their classification labels, where the goal is to predict whether an object is hazardous or not. The dataset includes various numerical and categorical features such as:
- **neo_id**: Unique Identifier for each Asteroid
- **name**: Name given by NASA
- **Absolute magnitude**: Describes intrinsic luminosity
- **estimated diameter min**: Minimum Estimated Diameter in Kilometres
- **estimated diameter max**: Maximum Estimated Diameter in Kilometres
- **orbiting body**: Planet that the asteroid orbits
- **relative velocity**: Velocity Relative to Earth in Kmph
- **miss distance**: Distance in Kilometres missed
- **Target Variable (is_hazardous)**: The boolean feature that shows whether an asteroid is harmful or not

The dataset was cleaned and preprocessed by handling missing values, removing duplicates, and encoding categorical variables.

## Approach

### Data Preprocessing
- **Handling Missing Values**: We filled missing values in numerical columns with the median and in categorical columns with the mode.
- **Feature Selection**: The dataset was cleaned by removing irrelevant columns (e.g., ID, object name, and orbiting body).
- **Balancing the Data**: We used SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes in the target variable (is_hazardous).

### Model Comparison
Four machine learning models were used to classify the data:
1. **Random Forest Classifier**
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **K-Nearest Neighbors (KNN)**

Each model was trained on the training data and evaluated on the test data. We calculated various performance metrics, including accuracy, precision, recall, and F1-score.

## Models Used
1. **Random Forest Classifier**: An ensemble learning method based on decision trees, suitable for handling large datasets and complex relationships.
2. **Logistic Regression**: A linear model used for binary classification tasks, assuming a linear relationship between input features and the log-odds of the target variable.
3. **Decision Tree Classifier**: A non-linear model splits data into subsets using feature values. It is easy to interpret and visualize but prone to overfitting.
4. **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm that classifies a sample based on the majority class of its nearest neighbors. It can be computationally expensive with large datasets.

## Evaluation Metrics
To evaluate the models, we used the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1-score**: The harmonic mean of precision and recall, providing a balanced measure.
- **Confusion Matrix**: A matrix showing the counts of true positive, true negative, false positive, and false negative predictions.

### Confusion Matrix Interpretation:
- **True Positives (TP)**: Correctly predicted hazardous objects.
- **True Negatives (TN)**: Correctly predicted non-hazardous objects.
- **False Positives (FP)**: Incorrectly predicted hazardous objects.
- **False Negatives (FN)**: Incorrectly predicted non-hazardous objects.

## Results and Comparison
The results were collected for all four models, including accuracy, precision, recall, F1-score, and confusion matrices. Here is a summary of the comparison:

| Model                   | Accuracy | Precision | Recall | F1-score |
|-------------------------|----------|-----------|--------|----------|
| Random Forest           | 0.96     | 0.96      | 0.96   | 0.96     |
| Logistic Regression     | 0.59     | 0.59      | 0.59   | 0.58     |
| Decision Tree           | 0.95     | 0.95      | 0.95   | 0.95     |
| KNN                     | 0.77     | 0.78      | 0.77   | 0.77     |

### Visualizations:
- **Confusion Matrix**: Visualized for each model to identify where the models are making errors.
- **Accuracy Comparison**: A bar plot comparing the accuracy scores for all models.

## Insights and Key Findings
- **Best Performing Model**: The Random Forest Classifier was the best-performing model with an accuracy of 96%. It also had the highest F1-score and recall, making it ideal for this classification task.
- **Model Complexity**: While the Logistic Regression model was simpler, it did not perform as well as the Random Forest due to the non-linear relationships in the data. The Decision Tree model was interpretable but slightly overfitted.

## Conclusion
Based on the results of this project, the Random Forest Classifier emerged as the most accurate and robust model for classifying hazardous Earth objects. It performed well across all evaluation metrics and provided a good balance of precision and recall. Future work could include hyperparameter tuning for better performance and exploring additional algorithms such as Gradient Boosting or XGBoost.

## Requirements
- Python 3.x
- scikit-learn
- pandas
- seaborn
- matplotlib
- imbalanced-learn

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```
## Installation and Usage
1. Clone the repository :
   ```bash
   git clone https://github.com/yourusername/ML-Model-Comparison.git
   cd ML-Model-Comparison
   ```
2. Run the Jupyter Notebook:
   ```bash
    jupyter notebook
   ```
3. Execute the model comparison: Follow the instructions in the notebook to load the dataset, preprocess the data, and train the models.

## Acknowledgments
The dataset used in this project was obtained from the NASA Near-Earth Objects dataset. Special thanks to the contributors of the machine learning and data science community for their resources and tutorials.



