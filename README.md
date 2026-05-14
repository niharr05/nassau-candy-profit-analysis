# Nassau Candy Profitability Dashboard

A small end-to-end analytics project that cleans and analyzes Nassau Candy sales data, then serves an interactive Streamlit dashboard for profitability and margin insights.

## Project Overview
- Step 1: `analysis.py` runs EDA, KPI summaries, charts, Pareto analysis, and K-Means clustering.
- Step 2: `app.py` launches a Streamlit dashboard with multiple tabs and filters.

## Features
- Product-level profitability rankings and contribution metrics
- Division performance comparison
- Pareto (80/20) analysis for top contributors
- K-Means clustering for product segmentation
- Interactive dashboard with filters and charts

## Repository Structure
```
.
├── Nassau Candy Distributor.csv
├── analysis.py
├── app.py
├── requirements.txt
├── data_processing.py
├── Executive_Summary.md
└── README.md
```

## Setup
1. Ensure Python is installed.
2. Install dependencies:

```
pip install -r requirements.txt
```

## Get the Project from GitHub
Option 1: Clone with Git
```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Option 2: Download ZIP
- Open the GitHub repo page
- Click the green "Code" button
- Choose "Download ZIP" and extract it

## Run Analysis (CLI)
```
python analysis.py
```

Outputs include:
- Console summaries and tables
- Charts saved as `nassau_charts.png`

## Run Dashboard (Streamlit)
```
streamlit run app.py
```

Open the app at: http://localhost:8501

## Data
Place the dataset in the project root with either of these filenames:
- `Nassau Candy Distributor.csv`
- `Nassau_Candy_Distributor.csv`

## Notes
- The dashboard includes filters for division, date range, and product search.
- K-Means clustering is based on profit and margin signals.

## Requirements
See `requirements.txt` for the full list of Python packages.