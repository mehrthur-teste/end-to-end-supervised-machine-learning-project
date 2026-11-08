# End-to-End Supervised Machine Learning Project

## Take-Home Group Assignment 1

| Item | Details |
| --- | --- |
| Type | Group Assignment |
| Weight | 20% of the Final Grade |
| Total Marks | 100 |
| Modules Covered | Modules 1–5 |
| Release | Week 4 |
| Due | Week 6 |

## Purpose

The purpose of this assignment is to apply the complete supervised machine learning workflow to a real-world classification problem. Working collaboratively, students will formulate a machine learning problem, explore and preprocess the dataset, develop and optimize classification models, evaluate their performance using appropriate metrics, interpret the results, and communicate their findings through a professional technical report and presentation.

This assignment emphasizes Course Learning Outcomes (CLOs) 1–4.

## Learning Outcomes

Upon successful completion of this assignment, students will be able to:

- Formulate a supervised machine learning classification problem.
- Perform exploratory data analysis and prepare datasets for machine learning.
- Develop, optimize, and compare multiple classification models.
- Evaluate model performance using appropriate evaluation metrics.
- Interpret results and justify model selection using quantitative evidence.
- Communicate technical findings effectively through written and oral presentations.

## Dataset

All groups will use the **IBM Telco Customer Churn Dataset**.

[Access the dataset on Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

## Requirements

Each group must develop and compare the following four models:

| Model | Algorithm | Configuration |
| --- | --- | --- |
| 1 | Logistic Regression | Baseline |
| 2 | Decision Tree | Baseline |
| 3 | Logistic Regression | Optimized |
| 4 | Decision Tree | Optimized |

The optimized models should be obtained by tuning one or more appropriate hyperparameters.

To ensure a fair comparison:

- Apply the same preprocessing procedures to all models.
- Use the same training and testing datasets for all models.
- Use identical evaluation metrics for every model.

## Assignment Tasks

### Part 1 — Problem Definition (10 Marks)

Clearly describe the machine learning problem. Include:

- Application domain.
- Business problem.
- Prediction objective.
- Target variable.
- Predictor variables.
- Why machine learning is an appropriate solution.

### Part 2 — Exploratory Data Analysis (15 Marks)

Perform an exploratory analysis of the dataset. Include:

- Dataset dimensions.
- Data types.
- Summary statistics.
- Missing values.
- Target class distribution.
- Whether the dataset is balanced or imbalanced.
- Correlation analysis, where appropriate.
- At least three meaningful visualizations.

**Interpret the findings rather than simply presenting figures.**

### Part 3 — Data Preprocessing (15 Marks)

Prepare the dataset for model development. Include, where appropriate:

- Handling missing values.
- Removing duplicate records.
- Removing irrelevant or identifier variables, if applicable.
- Encoding categorical variables.
- Feature scaling.
- Training/testing split.

**Each preprocessing decision must be briefly justified.**

### Part 4 — Model Development and Optimization (25 Marks)

Develop the following models:

1. Logistic Regression (Baseline).
2. Decision Tree (Baseline).
3. Optimized Logistic Regression.
4. Optimized Decision Tree.

For each model:

- Explain why it was selected.
- Describe the selected hyperparameters for optimized models.
- Train the model.
- Generate predictions using the testing dataset.
- Include well-documented Python code with appropriate comments.

### Part 5 — Model Evaluation and Comparison (20 Marks)

Evaluate every model using:

- Training Accuracy.
- Testing Accuracy.
- Classification Report.
- Confusion Matrix.

Present the results in a comparison table similar to the following. Dashes indicate results to be completed.

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-score | Comments |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (Baseline) | — | — | — | — | — | — |
| Decision Tree (Baseline) | — | — | — | — | — | — |
| Optimized Logistic Regression | — | — | — | — | — | — |
| Optimized Decision Tree | — | — | — | — | — | — |

Discuss:

- Which model performed best?
- Which model generalized better?
- Did optimization improve performance?
- Were there any trade-offs among the evaluation metrics?
- Which model would you recommend for deployment? Justify your recommendation.

### Part 6 — Discussion and Conclusion (15 Marks)

Discuss:

- Strengths of the recommended model.
- Limitations of the study.
- Did the optimized models outperform their corresponding baseline models? Explain why or why not.
- Possible improvements.
- Lessons learned during the project.
- Additional steps that should be completed before deploying the model in a real-world environment, such as additional validation, monitoring, periodic retraining, fairness assessment, or bias analysis.

**Support all conclusions using evidence from the experimental results.**

## Deliverables

Each group must submit:

| Deliverable | File | Details |
| --- | --- | --- |
| Jupyter Notebook | `GroupNumber.ipynb` | Project notebook |
| Technical Report | `GroupNumber.pdf` | Approximately 10 pages, excluding appendices |

### Suggested Report Structure

1. Title Page.
2. Problem Definition.
3. Exploratory Data Analysis.
4. Data Preprocessing.
5. Model Development.
6. Model Optimization.
7. Model Evaluation and Comparison.
8. Discussion.
9. Appendix (Optional).


## Run the Local Prediction API

The final cell of `Group4.ipynb` exports the selected fitted pipeline, including preprocessing, to `models/churn_pipeline.joblib`.

```bash
python3 -m pip install -r requirements.txt
python3 api/server.py
```

The API runs at `http://127.0.0.1:8000`. Send a customer record with:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @models/example_request.json
```

Edit the example JSON to enter the customer's data. See `api/README.md` for the input schema, endpoints and model compatibility details. This is a local inference demonstration; production deployment requires further validation and operational controls.


## Project Files

| File | Purpose |
| --- | --- |
| `Group4.ipynb` | Executed notebook with 20 predictors, four models, bootstrap, cost, calibration, threshold and permutation diagnostics |
| `ML_Group4_Report.docx` | Technical report with diagnostic appendices |
| `requirements-revised.txt` | Exact package versions for the integrated revision, also recorded in requirements.txt |
| `Group4.pdf` | Technical report for submission |
| `models/churn_pipeline.joblib` | Fitted CV-selected model, including preprocessing |
| `models/schema.json` | Input fields, categories and training versions |
| `models/example_request.json` | Example prediction request; ZIP Code is text |
| `api/server.py` | Local prediction API |

The feature review removed constant and repeated geographic fields and treats ZIP Code as a category. Model selection uses training-only stratified CV. The same test split was examined before this review, so the revised results should be confirmed on new independent data before deployment.

The November 8 revision was executed in full before synchronizing the report and model export. Bootstrap intervals and training diagnostics describe exploratory uncertainty and interpretation. The default prediction rule is retained. Baseline-tree results can vary across package versions. Use this notebook execution consistently for report tables.
