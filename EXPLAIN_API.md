# POST /explain (additive; `/assess` and the frozen contract v1.0.0 are unchanged)

Returns SHAP-style attributions for the four frozen models for one patient. Same request body as `/assess`; invalid input returns the **same 422 errors** as `/assess`.
It is intentionally not part of the frozen OpenAPI document (that document is pinned to `/assess` + `/meta` by the existing tests); this file is its specification.
Educational / decision-support use only. It describes the model's attribution, never causes of disease, and contains no clinical recommendations.

## Method (exact, additive, in log-odds space)
| Target | Model | Contribution of encoded feature j | Reference (log-odds) |
|---|---|---|---|
| CAD, LAD, RCA | logistic regression | `coef_j * (x_j - scaler_mean_j) / scaler_scale_j` (stored scaler) | intercept = a patient at the training mean of every feature |
| LCX | XGBoost | built-in TreeSHAP (`pred_contribs`) | the model's expected margin (TreeSHAP baseline) |

For every target: `reference + sum(contributions) = model log-odds`, and `sigmoid(...)` equals the probability `/assess` reports.
The server fails closed (HTTP 500, no body of attributions) if reconstruction and model output ever differ by more than 1e-6.
`/assess` rounds probabilities to 4 decimals; `estimate.probability_rounded` is that identical value, `estimate.probability` is the unrounded model output.

## Rows
The 54 encoded features are folded into the **50 form fields** (one-hot columns summed into their parent field, e.g. chest pain type, bundle branch block)
plus **one derived row** `bmi` ("BMI (derived from weight and height)", group `demographics`, `derived: true`). Rows are sorted by |contribution|.
`reference_value` is the training median (numbers) or most common value (categories/booleans) shown for context; contributions are measured from the model's own reference above.

## Response
```
{ explain_version, contract_version, model_version, request_id, assessment_id /* same hash as /assess */,
  targets: { CAD|LAD|LCX|RCA: {
      target, label, model_type, method: "exact_linear_logodds"|"treeshap",
      reference: { log_odds, probability, definition },
      estimate:  { log_odds, probability, probability_rounded, reconstructed_probability },
      contributions: [ { field, label, group, derived, unit, patient_value, display_value, reference_value, reference_display, reference_kind: "median"|"most_common",
                         contribution, direction: "positive"|"negative"|"neutral" } ],
      sections: [ { group, label, contribution } ] } },
  notes: [...], disclaimer }
```
