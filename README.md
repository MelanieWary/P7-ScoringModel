## Context of the project: 
Development of a loan scoring model that predicts the probability for a client to repay its loan and classifies the application accordingly (i.e. as loan granted or not granted)


## Objectives of the project:
### General objectives:
1. Build a scoring model to predict the probability that a client does not repay the loan he's applying for
2. Build an interactive dashboard, to be used by customer relationship managers, which displays the prediction, classification, interpretation of the prediction, client's data compared to those of other clients,...
   
### Detailed objectives:
1. Preprocess data (available [here](https://www.kaggle.com/c/home-credit-default-risk/data)) making use of Kaggle kernels adapted to the problematic (see P7_notebook_exploration_preprocessing.ipynb)
2. Test of different supervised binary classification models and methods to tackle unbalanced classes (see P7_notebook_modelling.ipynb)
3. Optimization of the previously selected as best model and method + threshold moving (see P7_notebook_modelling.ipynb)
4. Model deployment as API (see modelAPI.py)
5. Development and deployment of the dashboard (see dashboard.py)

## Outcomes:
Model API accessible [here](https://mw-loan-pred.herokuapp.com/docs#/default/predict_loan_repayment_prediction_post)

Dashboard accessible [here](https://share.streamlit.io/melaniewary/p7-scoringmodel/main/dashboard.py)