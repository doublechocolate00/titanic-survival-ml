# Titanic Survival Prediction

## Overview
A binary classification project that predicts passenger survival
on the Titanic using passenger attributes.

## Tech Stack
- Python
- pandas and NumPy
- scikit-learn
- MLflow
- Jupyter Notebook

## Workflow
1. Load and inspect the dataset.
2. Handle missing numerical and categorical values.
3. Scale numerical features and one-hot encode categorical features.
4. Engineer family-related features.
5. Train and evaluate Random Forest classifiers.
6. Track parameters, metrics, and models with MLflow.
7. Reload a logged model and verify prediction reproducibility.

## Selected Model
RandomForestClassifier with:
- n_estimators: 300
- max_depth: None
- min_samples_leaf: 2
- min_samples_split: 2
- random_state: 42

## Exploratory Evaluation
- Mean 5-fold CV F1: 0.7551
- CV F1 standard deviation: 0.0553
- Test F1: 0.7258
- Test accuracy: 0.8101

Note: The test set was evaluated repeatedly during experimentation.
These test metrics are exploratory and should not be treated as an
unbiased final estimate of generalization performance.

## Reproducibility
The trained pipeline was logged with MLflow, reloaded, and compared
against a freshly trained pipeline with the same configuration.
Their predictions on the test data were identical.

## Run the Project
1. Install the required Python packages.
2. Open the notebook and run its cells in order.
3. Start the MLflow UI to inspect tracked experiments.

## Future Improvements
- Use a dedicated validation strategy for model selection.
- Evaluate the final selected model on an untouched test set.
- Add automated tests and a prediction script.
