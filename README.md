# The Accuracy Rates of Weather Prediction in Istanbul: Exploring The Relationship Between Advancing Technological Meteorological Models and Climate Change 

**DSA 210 - Introduction to Data Science**  
**Spring-2026**  
**Sabancı University**  
**Student / ID: Dilay Sena Cenk / 33870**

---

**🌐 Project Presentation Website:** [https://dsa210project-dilaysenacenk.lovable.app](https://dsa210project-dilaysenacenk.lovable.app)

---

## Project Overview & Motivation
This project investigates the critical intersection between advancing meteorological predictive technologies and rising climate volatility, using Istanbul as a primary case study. Inspired by the stark contradiction observed during the January 2024 snow warning—where extensive preemptive city-wide shutdowns were met with no actual snowfall—the research explores whether technological forecasting is failing to keep pace with climate shifts, or if it instead drives preventive systemic resilience. 

While the initial hypothesis assumed that rapid climate volatility would outpace predictive capabilities, the empirical findings demonstrate a **"catalyst effect"**: climate stress (temperature anomalies) directly acts as a structural driver for technological investment, adaptation, and market scaling.

---

## Data Infrastructure
To ensure scientific rigor and data reproducibility, the analysis utilizes highly reliable, multi-decadal professional datasets:
* **ECMWF ERA5 Reanalysis:** Historical, high-resolution global atmospheric parameters.
* **CDS API (Climate Data Store):** Programmatic pipelines for automated, reproducible climate variable retrieval.
* **Weather Underground:** Localized, ground-level daily observations for cross-reference and validation.
* **Macro Technological Indices:** Aggregated market indicators tracking tech-sector adaptation and investment shifts.

---

## Data Pipeline & Processing
Instead of processing flat, continuous historical lines, the project implements an **Extreme Event Sampling** strategy:
* **Statistical Thresholding:** Decades of ERA5 temperature data were filtered using standard deviations ($\pm2$ SD from the seasonal mean) to systematically isolate periods of severe climate shocks (e.g., sudden heatwaves or acute frosts).
* **Temporal Synchronization:** Granular hourly and daily ERA5 inputs were mathematically downsampled to achieve perfect temporal alignment with tech market movement indicators.

---

## Methodological Stages
* **Exploratory Data Analysis (EDA):** Conducted long-term structural break analyses and multi-variable correlations between climate anomaly intervals and tech-market indicators.
* **Statistical Inference:** Deployed **Two-sample t-tests** to quantify systemic market shocks before and after severe weather anomalies, and **Pearson Correlation** to measure linear baseline dependencies.
* **Predictive Modeling:** Built, optimized, and evaluated **Linear Regression, Decision Trees**, and **Random Forest** architectures to model tech-sector adaptive growth.
* **Feature Engineering:** Developed multi-day lag indicators (e.g., `Temp_Lag_7`, `Temp_Lag_30`) via the CDS API to successfully capture the market’s operational "memory" and delayed adaptation thresholds.

---

## Executive Summary of Findings
* **Structured Linearity:** The predictive performance of the models reached an exceptionally high $R^2$ score of 0.99. The baseline Linear Regression matched the performance of more complex ensemble methods (Random Forest), confirming that the relationship between environmental volatility and technological adaptation scales in a highly predictable, structured linear fashion.
* **Error Reliability:** Residual and error distribution analyses confirmed a healthy, homoscedastic "white noise" scattering along the zero-bias line, validating that the predictive pipeline remains statistically stable even under extreme climatic disruptions.
* **Market Memory:** Feature importance analysis proved that time-lagged temperature features exert the heaviest weight on the model. This confirms that tech adaptation operates with a structured time delay, meaning the market actively prices in environmental stress based on past observations rather than instantaneous reactions.

---

## Limitations & Future Horizons
* **Threshold Restrictions:** Relying strictly on a $\pm2$ standard deviation filter limits the effective data to acute shock windows, masking subtler, cumulative climate trends.
* **Scale Asymmetry:** Merging localized grid-based spatial ERA5 climate cells with macro-level economic indices causes a smoothing effect that might obscure regional, micro-climate infrastructure impacts.
* **Linearity Boundary Conditions:** While the current 2023–2025 timeframe exhibits strong linearity, the model assumes infinite adaptability and does not yet account for systemic tipping points where catastrophic climate failures could break market correlation.
* **Next Steps:** Future extensions aim to integrate real-time digital panic indicators (Google Trends and scraped social media sentiment using BERT models), deploy non-linear deep learning time-series architectures, and introduce macroeconomic controls (CPI, interest rates).

---

## Academic Integrity
This project is an original, individual work developed for **DSA 210**. AI tools (Gemini) were strictly leveraged for code refactoring, structural writing assistance, and debugging support, fully adhering to Sabancı University’s academic integrity policies.
