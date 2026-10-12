# Track A /assess contract v1.0.0 (DRAFT FOR REVIEW)

Planned endpoints: `GET /schema` (features.json) | `GET /model-card` | `POST /assess` (this contract) | `POST /explain` (later, same request body + `target`).

## Request
`POST /assess` body: `{ "request_id"?: string, "patient": { ...50 required fields... } }`. Unknown keys are rejected. Booleans must be JSON true/false. No imputation: a missing field is an error.

| Group | Field (type; allowed values / unit) |
|---|---|
| demographics | `age` number (years); `sex` enum male/female; `weight_kg` number (kg); `height_cm` number (cm) |
| history_risk_factors | `diabetes` bool; `hypertension` bool; `current_smoker` bool; `ex_smoker` bool; `family_history` bool; `dyslipidemia` bool; `chronic_renal_failure` bool; `cerebrovascular_accident` bool; `airway_disease` bool; `thyroid_disease` bool; `congestive_heart_failure` bool |
| exam_symptoms | `bp_mmhg` number (mmHg); `pulse_rate_bpm` number (bpm); `edema` bool; `weak_peripheral_pulse` bool; `lung_rales` bool; `systolic_murmur` bool; `diastolic_murmur` bool; `chest_pain_type` enum typical/atypical/nonanginal/none; `dyspnea` bool; `function_class` int [0, 1, 2, 3]; `low_threshold_angina` bool |
| ecg | `q_wave` bool; `st_elevation` bool; `st_depression` bool; `t_inversion` bool; `lvh` bool; `poor_r_progression` bool; `bundle_branch_block` enum none/lbbb/rbbb |
| labs | `fbs` number (mg/dL); `creatinine` number (mg/dL); `tg` number (mg/dL); `ldl` number (mg/dL); `hdl` number (mg/dL); `bun` number (mg/dL); `esr` number (mm/h); `hemoglobin` number (g/dL); `potassium` number (mEq/L); `sodium` number (mEq/L); `wbc` number (cells/uL); `lymphocyte_pct` number (%); `neutrophil_pct` number (%); `platelets` number (x10^3/uL) |
| echo | `ef_tte_pct` number (%); `rwma_region` int [0, 1, 2, 3, 4]; `valvular_heart_disease` enum none/mild/moderate/severe |

Never accepted as input (rejected with a specific code): `LAD`, `LCX`, `RCA`, `Cath`, `CAD` (forbidden_target_field); `bmi`, `obesity` (derived_field_not_accepted; BMI is computed from weight_kg and height_cm).

### Example request (training row, for format only)
```json
{
 "request_id": "median_cad_row",
 "patient": {
  "age": 56,
  "sex": "female",
  "weight_kg": 62,
  "height_cm": 155,
  "diabetes": false,
  "hypertension": true,
  "current_smoker": false,
  "ex_smoker": false,
  "family_history": true,
  "dyslipidemia": true,
  "chronic_renal_failure": false,
  "cerebrovascular_accident": false,
  "airway_disease": false,
  "thyroid_disease": false,
  "congestive_heart_failure": false,
  "bp_mmhg": 130,
  "pulse_rate_bpm": 75,
  "edema": false,
  "weak_peripheral_pulse": false,
  "lung_rales": false,
  "systolic_murmur": false,
  "diastolic_murmur": false,
  "chest_pain_type": "typical",
  "dyspnea": false,
  "function_class": 2,
  "low_threshold_angina": false,
  "q_wave": false,
  "st_elevation": false,
  "st_depression": false,
  "t_inversion": false,
  "lvh": false,
  "poor_r_progression": false,
  "bundle_branch_block": "none",
  "fbs": 88,
  "creatinine": 1.3,
  "tg": 100,
  "ldl": 180,
  "hdl": 64.0,
  "bun": 18,
  "esr": 18,
  "hemoglobin": 13.3,
  "potassium": 4.1,
  "sodium": 140,
  "wbc": 4900,
  "lymphocyte_pct": 40,
  "neutrophil_pct": 51,
  "platelets": 218,
  "ef_tte_pct": 55,
  "rwma_region": 0,
  "valvular_heart_disease": "mild"
 }
}
```

## Response 200
```json
{
 "contract_version": "1.0.0",
 "model_version": "v1.0.0",
 "request_id": "median_cad_row",
 "assessment_id": "736d0771ddc629e58fa58f737250c7664ad8f08abd1a27bdc5489a82353f4922",
 "predictions": {
  "CAD": {
   "target": "CAD",
   "label": "Overall CAD",
   "probability": 0.8911,
   "model_type": "logistic_regression",
   "definition": "Dataset overall CAD label (Cath = CAD). In the source data this equals >=50% narrowing in at least one of LAD/LCX/RCA for 302 of 303 patients."
  },
  "LAD": {
   "target": "LAD",
   "label": "LAD stenosis",
   "probability": 0.6195,
   "model_type": "logistic_regression",
   "definition": "Whole-vessel dataset label for LAD (>=50% narrowing). A vessel-level probability only; it carries NO information about lesion location."
  },
  "LCX": {
   "target": "LCX",
   "label": "LCX stenosis",
   "probability": 0.2682,
   "model_type": "xgboost",
   "definition": "Whole-vessel dataset label for LCX (>=50% narrowing). A vessel-level probability only; it carries NO information about lesion location."
  },
  "RCA": {
   "target": "RCA",
   "label": "RCA stenosis",
   "probability": 0.417,
   "model_type": "logistic_regression",
   "definition": "Whole-vessel dataset label for RCA (>=50% narrowing). A vessel-level probability only; it carries NO information about lesion location."
  }
 },
 "model_performance": {
  "CAD": {
   "cv_roc_auc": 0.9252,
   "discrimination_tier": "high",
   "...": "(full block in real response)"
  },
  "LAD": {
   "cv_roc_auc": 0.8295,
   "discrimination_tier": "moderate",
   "...": "(full block in real response)"
  },
  "LCX": {
   "cv_roc_auc": 0.7337,
   "discrimination_tier": "limited",
   "...": "(full block in real response)"
  },
  "RCA": {
   "cv_roc_auc": 0.7124,
   "discrimination_tier": "limited",
   "...": "(full block in real response)"
  }
 },
 "consistency": {
  "vessels_exceeding_cad": [],
  "margin": 0.1,
  "probabilities_modified": false,
  "note": "Targets are modeled independently; a vessel probability above CAD is flagged, never adjusted."
 },
 "input_checks": {
  "derived": {
   "bmi": 25.81
  },
  "out_of_observed_range": [],
  "sparse_categories": [],
  "extrapolation_flag": false
 },
 "uncertainty": null,
 "data_support": null,
 "population_note": "Probabilities are conditional on the training population (patients referred for angiography; 71% CAD prevalence) and are not general-population risk.",
 "disclaimer": {
  "short": "Decision support / educational use only. Not a diagnosis and not a substitute for formal diagnostic imaging.",
  "text": "These estimates come from a statistical research model trained on 303 patients from a single referral cohort. They are for decision-support and educational purposes only. They are not a diagnosis, do not show where a blockage is, and are not a substitute for formal diagnostic imaging (e.g., coronary angiography or CT angiography) or evaluation by a clinician.",
  "required_display": true
 }
}
```

## Error 422
`{ "errors": [ {"field": "patient.age", "code": "missing_field", "message": "..."} ] }` with codes: missing_field, unknown_field, forbidden_target_field, derived_field_not_accepted, invalid_type, invalid_enum, not_finite, invalid_value.

## Semantics
- `probability`: raw model output (rounded to 4 d.p.), never calibrated post-hoc, never adjusted for display. No hard yes/no labels are returned.
- `consistency.vessels_exceeding_cad`: warning only. The 0.10 margin is a display heuristic, NOT a clinical or statistical threshold; probabilities are never modified, clamped or normalized. Targets are independent models.
- `input_checks` (data-support information, NOT prediction uncertainty): deterministic checks against training-set ranges and category counts. `extrapolation_flag` is true if any numeric input (or derived BMI) lies outside the observed training range.
- `uncertainty`, `data_support`: reserved (null in v1.0.x); additive in a minor version.
- `assessment_id`: sha256(model_version + encoded 54-vector); lets /explain and the UI bind an explanation to the exact assessment (stale-response guard).
- `disclaimer.required_display` is always true: clients must render `disclaimer.short` persistently on the main view.

- `rwma_region` / `function_class`: integer codes exactly as recorded in the dataset; their meanings are not documented in the provided files, so none is assigned. The models use them as plain numbers.
- Backend endpoints: `POST /assess`, `GET /meta` (frozen model-card performance + limitations, verbatim).
