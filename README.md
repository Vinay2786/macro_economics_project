# ML-Based Analysis and Prediction of Investment Response to GDP Changes Using the Accelerator Principle

An undergraduate Macroeconomics empirical machine learning project investigating whether changes in India's real GDP explain and estimate real Gross Fixed Capital Formation (GFCF) based on World Bank data (1990–2025).

---

## 1. Project Overview & Research Objective

The objective of this research is to evaluate the empirical applicability of the **Accelerator Principle of Investment** using annual macroeconomic data for India. Specifically, we investigate whether changes in India's real Gross Domestic Product ($\Delta\text{GDP}$) can be used to estimate real Gross Fixed Capital Formation ($\text{GFCF}$) using a linear machine learning model.

- **Theoretical Foundation:** The Accelerator Principle (Clark, 1917; Samuelson, 1939).
- **ML Algorithm:** Linear Regression (`scikit-learn` primary predictive model; `statsmodels` OLS for statistical inference).
- **Country & Data Source:** India, World Bank national accounts data.
- **Currency & Units:** Constant 2015 US Dollars ($).
- **Timeline:** 1990 (Base year), 1991–2017 (Training period, $N=27$), 2018–2025 (Held-out testing period, $N=8$).

---

## 2. Theoretical Framework & Model Specification

### Basic Accelerator Principle (Theoretical Form)
In macroeconomic theory, the simple accelerator model posits that net induced investment ($I_{net}$) is proportional to the rate of change in aggregate output:

$$I_{net}(t) = v \times [GDP(t) - GDP(t-1)]$$

Where:
- $I_{net}(t)$: Net induced investment in year $t$.
- $v$: Accelerator coefficient (capital-output ratio requirement).
- $GDP(t) - GDP(t-1)$: Change in real GDP from the previous period ($\Delta GDP(t)$).

### Empirical Regression Model
Because national accounts data record **Gross Fixed Capital Formation** ($\text{GFCF}$)—which includes replacement investment for capital depreciation—we estimate the empirical regression model:

$$\text{GFCF}(t) = \alpha + \beta \times \Delta\text{GDP}(t) + \varepsilon(t)$$

Where:
- $\text{GFCF}(t)$: Current-year real Gross Fixed Capital Formation.
- $\Delta\text{GDP}(t)$: Current-year change in real GDP from the previous year.
- $\alpha$: Regression intercept (capturing baseline replacement and autonomous investment).
- $\beta$: Estimated empirical slope coefficient.
- $\varepsilon(t)$: Stochastic error term representing all unobserved economic influences (interest rates, credit availability, business expectations, government policies).

### Academic Distinctions & Caveats
1. **Gross vs. Net Investment:** GFCF measures gross fixed investment and includes depreciation replacement; it is not identical to theoretical net induced investment.
2. **Empirical Slope vs. Accelerator Coefficient:** The regression coefficient $\beta$ represents an empirical slope, not strictly the theoretical parameter $v$.
3. **Association vs. Causation:** The regression model identifies empirical association and does not establish structural causality.
4. **Contemporaneous Estimation vs. Advance Forecasting:** Both $\Delta\text{GDP}(t)$ and $\text{GFCF}(t)$ pertain to the same year $t$. Hence, this model performs a **same-period estimation / held-out period evaluation**, not a true advance forecast made prior to observing that year's GDP growth.

---

## 3. Dataset & Rigorous Validation

The dataset is sourced from `India_GDP_GFCF_Model_Data_1990_2025.xlsx` (Sheet: `Model data`). 
Prior to modeling, nine validation checks were performed:

1. **Unique Years:** Each year from 1990 to 2025 appears exactly once.
2. **Coverage:** All 36 years (1990–2025) are present.
3. **Data Types:** All GDP and GFCF observations are strictly numeric.
4. **Positivity:** GDP and GFCF are strictly positive for all years.
5. **Delta Calculation:** $\Delta\text{GDP}(t) = \text{GDP}(t) - \text{GDP}(t-1)$ for all $t \ge 1991$.
6. **Floating-Point Agreement:** Uploaded $\Delta\text{GDP}$ values match calculated differences with maximum discrepancy $0.0000 \times 10^0$.
7. **Base Year Handling:** The 1990 $\Delta\text{GDP}$ is appropriately recorded as `NaN` (1989 is unavailable).
8. **No Missing Target Data:** Zero missing predictor or target values exist across 1991–2025.
9. **Split Alignment:** Exactly 1 Base year (1990), 27 Training years (1991–2017), and 8 Testing years (2018–2025).

A cleaned version is saved to `data/india_gdp_gfcf_clean.csv`.

---

## 4. Empirical Model Estimation & Verification

The model was fitted strictly on the 1991–2017 training observations ($N=27$).

### Estimated Parameters
- **Intercept ($\alpha$):** $\$31,478,301,706.63$ ($\approx \$31.48$ Billion Constant 2015 USD)
- **Slope ($\beta$):** $3.911737$
- **Mathematical Equivalence:** The coefficients obtained from `scikit-learn LinearRegression` and `statsmodels OLS` match within machine tolerance (difference $< 10^{-4}$ on intercept, $< 10^{-15}$ on slope).

### Model Equation
$$\widehat{\text{GFCF}}(t) = 31.48\text{ Billion} + 3.9117 \times \Delta\text{GDP}(t)$$

---

## 5. Performance Evaluation

| Period | Observations | MAE (USD) | MAE (Billion USD) | RMSE (USD) | RMSE (Billion USD) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (1991–2017)** | 27 | $\$63,830,519,520$ | **$\$63.83\text{B}$** | $\$80,568,701,572$ | **$\$80.57\text{B}$** | **$0.8264$** |
| **Testing (2018–2025)** | 8 | $\$302,721,907,120$ | **$\$302.72\text{B}$** | $\$494,988,596,848$ | **$\$494.99\text{B}$** | **$-11.6864$** |

### Understanding the Test Metrics
- **Training Fit:** The in-sample $R^2$ of $0.8264$ demonstrates that over 82.6% of the variation in GFCF during 1991–2017 was tracked by the simple accelerator model.
- **Negative Test $R^2$:** In out-of-sample evaluation, $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. A negative value indicates that historical pre-2018 parameters underperformed relative to a naive horizontal mean prediction. This result is driven entirely by the unprecedented macroeconomic contraction during the 2020 COVID-19 pandemic shock.

---

## 6. Stochastic Error Term vs. Residuals (Faculty Emphasis)

### Theoretical Distinction
- **Error Term $\varepsilon(t)$:** An unobservable, theoretical random variable representing all non-modeled macroeconomic influences in the true data generating process.
- **Residual $e(t) = y(t) - \hat{y}(t)$:** The observed algebraic difference between actual and fitted values computed after model estimation.

### Diagnostic Testing on Training Residuals ($N=27$)
1. **Residual Mean:** $3.7864 \times 10^{-5}\text{ USD}$ (Zero conditional mean holds mathematically via OLS intercept).
2. **Residual Standard Deviation:** $\$82.10\text{ Billion USD}$.
3. **Durbin-Watson Test:** $\text{DW} = 1.2881$ ($< 2.0$), indicating moderate positive autocorrelation, typical in macroeconomic time series due to cyclical inertia.
4. **Shapiro-Wilk Normality Test:** $W = 0.8842$, $p = 0.0059$. Since $p < 0.05$, the null hypothesis of normality is rejected at the 5% significance level, showing sample skewness.
5. **Breusch-Pagan Heteroskedasticity Test:** $\text{LM} = 0.0371$, $p = 0.8473$. Since $p > 0.05$, we fail to reject the null hypothesis of homoskedasticity (no severe heteroskedasticity detected).

*Academic Caution:* Because the sample consists of 27 annual observations, these diagnostic findings must be interpreted tentatively.

---

## 7. COVID-19 Period Analysis (2018–2025)

| Year | Real GDP Change ($\Delta\text{GDP}$) | Actual GFCF | Predicted GFCF | Residual ($y - \hat{y}$) | Absolute Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2018 | $+\$156.96\text{B}$ | $\$785.87\text{B}$ | $\$645.46\text{B}$ | $+\$140.41\text{B}$ | $\$140.41\text{B}$ |
| 2019 | $+\$100.23\text{B}$ | $\$794.88\text{B}$ | $\$423.55\text{B}$ | $+\$371.33\text{B}$ | $\$371.33\text{B}$ |
| **2020** | **$-\$155.37\text{B}$** | **$\$738.48\text{B}$** | **$-\$576.31\text{B}$** | **$+\$1,314.79\text{B}$** | **$\$1,314.79\text{B}$** |
| 2021 | $+\$245.52\text{B}$ | $\$867.87\text{B}$ | $\$991.88\text{B}$ | $-\$124.01\text{B}$ | $\$124.01\text{B}$ |
| 2022 | $+\$211.49\text{B}$ | $\$941.16\text{B}$ | $\$858.77\text{B}$ | $+\$82.39\text{B}$ | $\$82.39\text{B}$ |
| 2023 | $+\$215.65\text{B}$ | $\$1,009.42\text{B}$ | $\$875.03\text{B}$ | $+\$134.39\text{B}$ | $\$134.39\text{B}$ |
| 2024 | $+\$227.64\text{B}$ | $\$1,074.13\text{B}$ | $\$921.94\text{B}$ | $+\$152.19\text{B}$ | $\$152.19\text{B}$ |
| 2025 | $+\$259.85\text{B}$ | $\$1,150.20\text{B}$ | $\$1,047.94\text{B}$ | $+\$102.27\text{B}$ | $\$102.27\text{B}$ |

### Empirical Insights
1. **The 2020 Anomaly:** India's real GDP contracted by $-\$155.37\text{B}$ in 2020 due to national lockdown measures. Because the model slope is $+3.9117$, the linear formula extrapolated a negative GFCF ($-\$576.31\text{B}$). However, actual GFCF remained positive at $+\$738.48\text{B}$ because gross capital formation cannot be negative in practice.
2. **Error Magnitudes:** The 2020 prediction error of $\$1,314.79\text{B}$ is roughly 10 times larger than the average post-2020 error.
3. **Post-Pandemic Normalization:** From 2021 to 2025, errors quickly returned to a stable band between $\$82\text{B}$ and $\$152\text{B}$, showing that the simple accelerator resumes sensible tracking during sustained expansion phases.

---

## 8. Project Structure

```text
MacroEconomics_Accelerator_Project/
├── India_GDP_GFCF_Model_Data_1990_2025.xlsx  # Original World Bank input dataset
├── app.py                                   # Streamlit presentation dashboard
├── model.py                                 # Validation, modeling, metrics, and diagnostics
├── requirements.txt                         # Required Python dependencies
├── README.md                                # Project documentation and report
├── data/
│   └── india_gdp_gfcf_clean.csv             # Cleaned and validated dataset
└── outputs/
    ├── graph1_gfcf_vs_delta_gdp.png         # Graph 1: Actual GFCF vs Change in GDP
    ├── graph2_gfcf_vs_year_training.png     # Graph 2: Actual vs Predicted GFCF (1991–2017)
    ├── graph3_gfcf_vs_year_testing.png      # Graph 3: Actual vs Predicted GFCF (2018–2025)
    ├── graph4_error_term_vs_year.png        # Graph 4: Regression Error Term vs Year (1991–2025)
    ├── predictions.csv                      # Full prediction dataset
    ├── model_metrics.csv                    # Train and test MAE, RMSE, R-squared
    ├── diagnostic_summary.csv               # Statistical tests on residuals
    └── covid_analysis.csv                   # 2018–2025 detailed performance
```

---

## 9. How to Run the Project

### Environment Setup
If not already activated, activate your virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Or install the dependencies:
```bash
pip install -r requirements.txt
```

### Running the Python Model Script
To run the automated data validation, fit the models, calculate all metrics, and generate output graphs:
```bash
python model.py
```
*(Or with venv explicitly: `.\.venv\Scripts\python.exe model.py`)*

### Launching the Streamlit Dashboard
To launch the minimalist presentation dashboard:
```bash
streamlit run app.py
```
*(Or with venv explicitly: `.\.venv\Scripts\streamlit.exe run app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.
