# Local churn prediction API

The final notebook cell exports the fitted CV-selected model and its preprocessing to `models/churn_pipeline.joblib`. The current model uses 19 fields and excludes CLTV, constant fields and repeated geographic representations. ZIP Code is excluded in this revision. Optimized Logistic Regression uses a fixed probability cutoff of 0.60, embedded in the exported classifier and recorded as decision_threshold in the schema. Run all notebook cells before using a newly trained model.

From the repository root, use the same Python environment that trained the model:

```bash
python3 -m pip install -r models/requirements-inference.txt
python3 api/server.py
```

The server listens locally at `http://127.0.0.1:8000`.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/schema
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @models/example_request.json
```

Edit `models/example_request.json` with the customer's values. `/schema` lists exact field names, types and observed training categories. Supply all 19 fields; `null` uses the pipeline's imputation. Do not send ZIP Code, CustomerID, CLTV or outcome columns. Numerical values must be finite JSON numbers. Unknown text categories are handled by the fitted encoder.

A successful response contains `prediction` (1 for churn, 0 otherwise), `label`, `churn_probability` (0 to 1) and `model`. Invalid requests return HTTP 400. The probability is an uncalibrated model estimate, not a guarantee or an estimate of campaign success.

This is a local demonstration API. Deployment requires authentication, operational monitoring and further model validation. Load only model files you trust. Model serialization depends on package versions: `models/schema.json` records the exact training versions. Recreate the artifact in your environment rather than loading it with incompatible scikit-learn versions.
