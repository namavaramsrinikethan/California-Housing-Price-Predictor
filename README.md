# California Housing Price Predictor
Maincrafts Technology | AI/ML Internship Task 1 | N. Srinikethan

A linear regression model predicting historical California census block-group median house values. The notebook includes EDA, median imputation, an 80/20 split (seed 42), a mean baseline and residual diagnostics.

| Test metric | Linear regression | Mean baseline |
|---|---:|---:|
| MAE | $53,320 | $90,607 |
| RMSE | $74,558 | $114,486 |
| R² | 0.5758 | approximately 0 |

## Files
- `task1_ml_linear_regression.ipynb`: executed notebook with explanatory comments.
- `task1_ml_linear_regression_report_refined.pdf`: final four-page report.
- `house_price_model.joblib`: trained pipeline (scikit-learn 1.6.1).
- `app.py`: optional Streamlit prediction interface.
- `requirements.txt`: pinned app dependencies.

## Run locally
Use Python 3.11 or 3.12 in a virtual environment. From this folder:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The default input should yield about $184,831. The app requires all eight features and converts income to the training units ($10,000) and the model output to USD. Only load trusted joblib files.

## Notebook
Run in Colab or install Jupyter, matplotlib and seaborn alongside the app dependencies. The final cell generates the original automated report; the separately refined PDF is the intended submission report.

## Limitations
Historical 1990 data, capped target, a random rather than spatial split and negative/extrapolated predictions. Educational use only; not a current real-estate appraisal.

## Submission
The assignment requests the notebook and 2–4-page PDF; saved model and UI are optional. Follow the organizer's separate submission instructions. A public repository or deployed app is not required by the PDF. Review documents for personal information before publishing publicly.
