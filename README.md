# SHAP Explainer: Feature Importance

This ModelOp Center monitor computes **Feature Importance** metrics using a pre-trained **SHAP Explainer**.

## Input Assets

| Type | Number | Description |
| ------ | ------ | ------ |
| Baseline Data | **0** | |
| Sample Data | **1** |  A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has 
     - a **pre-trained SHAP Exaplainer** asset (as a `.pickle` file)
     - a list of **predictive features** as used by the scoring model (as a `.pickle` file)

## Execution
1. `init` function loads the list of predictive features and the SHAP explainer from `.pickle` files.
2. `metrics` function pre-processes input data to get dummy variables, then computes feature importance by computing SHAP values.
3. Test results are appended to the list of `interpretability` tests to be returned by the model.

## Monitor Output

```JSON
{
    "interpretability": [
        {
            "test_name": "SHAP",
            "test_category": "interpretability",
            "test_type": "shap",
            "metric": "feature_importance",
            "test_id": "interpretability_shap_feature_importance",
            "values": {
                <feature_1>:<feature_1_importance>,
                <feature_2>:<feature_2_importance>,
                ...:...
            }
        }
    ]
}
```