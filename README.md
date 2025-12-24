# Finance ML Project — Market Return Prediction & Visual EDA

This repository explores whether historical market data can be used to predict future movements using a small end-to-end pipeline: data loading → cleaning → feature engineering → correlation analysis → return prediction → visualisation.

**Core takeaway:** history cannot reliably predict market movements, but models may capture a *weak directional structure* in returns. Evaluation is done using regression metrics such as **MSE** and **R²**, alongside visual inspection of **actual vs predicted** returns.

---

## Project Status (Important)

⚠️ **This repository is a work in progress.**  
The overall organisation is currently incomplete and still being refactored.

- **`model.py`** focuses on training + generating predictions  
- **`visualisation.py`** focuses on charts/EDA (correlation matrix, candlesticks + indicators, prediction plots)

✅ **They must be run separately** from inside the `src/` folder.

---

## Requirements

- **Python 3.14**
- Packages used (typical):
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `mplfinance`
  - `scikit-learn`

> Note: if you are using Homebrew Python on macOS, installs may require  
> `--break-system-packages` due to PEP 668.

Example:
```bash
python3 -m pip install --break-system-packages pandas numpy matplotlib seaborn mplfinance scikit-learn
