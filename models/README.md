# Model artifacts

| File | In GitHub | In Google Drive zip |
|------|-----------|---------------------|
| `bank_classification_model.joblib` | Yes (small) | Yes |
| `housing_regression_model.joblib` | No (>100 MB GitHub limit) | Yes |
| `*_metrics.json` | Yes | Yes |

The public Streamlit app uses **SageMaker endpoints** for cloud inference. To train the regression model locally:

```bash
python training/train_regression.py
```

The large `.joblib` file is included in the submission zip from `./scripts/package_submission.sh`.
