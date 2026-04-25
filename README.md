# Flight Delay Analysis – Phase 1

## 📌 Project Overview

This project analyzes flight delays for flights departing from New York State airports.
The goal is to explore patterns in arrival delays (ArrDel15) using exploratory data analysis (EDA).

The dataset includes monthly flight records from December 2024 to October 2025 obtained from the Bureau of Transportation Statistics (BTS).

## Data Source

This project uses the U.S. DOT Bureau of Transportation Statistics
On-Time Performance dataset (1987–Present)
-Following is the link to this datasets:-

https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr

The following monthly datasets were used:

- December 2024  
- January 2025  
- February 2025  
- March 2025  
- April 2025  
- May 2025  
- June 2025  
- July 2025  
- August 2025  
- September 2025  
- October 2025
- November 2025 


-Step 1:-
Select "All" in Filter Geography

-Step2:-
Select Filter Year=2025 

-Step3:-
Select Filter month = 'January"

-Step 4:-
Select "prezipped files" and then press download

January Dataset will be downloaded
Repeat this process for all the months

-NOTE:- Dec 2025 data is not available so I have used Dec 2024 Dataset instead, So once done downloading Jan-Nov 2025 dataset, Download Dec 2024 dataset by selecting Filter Year= 2024 and Filter Month = November




-Download  this monthly BTS On-Time Performance CSV files and place them inside:

data/raw/

The project will not run without raw data files.

---

## 📂 Repository Structure
flight-delay-project/
│
├── README.md
├── requirements.txt
├── data/
│ ├── raw/ # Original BTS CSV files
│ └── processed/ # Cleaned datasets
├── outputs/
│ └── eda/ # EDA tables and plots
└── src/
├── data_collection.py
├── data_cleaning.py
└── eda.py

---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment

python3 -m venv .venv
source .venv/bin/activate
### 2️⃣ Install dependencies
pip install -r requirements.txt

### ▶️ How to Run the Project

### Step 1: Combine Raw Data
### Place all original BTS CSV files inside:
data/raw/
### then run:
python src/data_collection.py

### Step 2: Clean and Process Data
python src/data_cleaning.py
### This generates the cleaned dataset inside:
data/processed/

### Step 3: Run Exploratory Data Analysis
python src/eda.py
### EDA outputs (tables and plots) are saved in:
outputs/eda/

# Flight Delay Prediction (Phase 2)

This project predicts whether a flight will arrive delayed using machine learning.

## Dataset

The dataset contains flight information combined with weather data.

## Pipeline

1. Data collection
2. Data cleaning
3. Feature engineering
4. Machine learning models
5. Model evaluation
6. MCP deployment

## Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Naive Bayes
- SVM

## Best Model

XGBoost achieved the best performance with the highest ROC-AUC score.

## MCP Deployment

The best performing model (XGBoost) was deployed using an MCP server.

Run locally:
python src/mcp/server.py

The server exposes a `predict` tool that returns flight delay predictions.

## Requirements

Install dependencies:

pip install -r requirements.txt

Phase 3 — Databricks / Spark / Delta Lake

Phase 3 re-implements the Phase 2 analytics on Apache Spark with Delta Lake using the Medallion (Bronze → Silver → Gold) architecture on Databricks Free Edition. It also integrates Meteostat hourly weather as an additional data source.

Weather join match rate: 99.82%. Full writeup in reports/Phase3_summary.pdf.

Notebooks
All in notebooks/databricks/:
01_bronze_flights        → bronze_flights
02_silver_flights        → silver_flights
03_gold_flights          → gold_flight_features + business aggregates
04_mllib_models          → LR / RF / GBT + MLflow
05_weather_pipeline      → Meteostat bronze/silver/gold
06_joined_analysis       → joined model + 3 insights

How to Reproduce in Databricks

Sign up for Databricks Free Edition: https://www.databricks.com/learn/free-edition
Upload the data: In Catalog → workspace.default, create a Volume named flight_data and upload data/processed/ny_flights_phase3_bronze.csv to it.
Import notebooks: Workspace → create folder NY_flight_delay_phase3 → Import all six .ipynb files from notebooks/databricks/.
Run in order (each notebook reads the previous one's Delta tables): 01 → 02 → 03 → 04 → 05 → 06. Click Run all on each, with Serverless attached.

Serverless-specific notes

Notebook 04 calls 
%pip install xgboost then dbutils.library.restartPython(). 
After the restart, run cells from the top — don't use "Run all" again immediately.
SPARKML_TEMP_DFS_PATH is set to a UC Volume path inside Notebooks 04 and 06. Required for MLlib on serverless.
CrossValidator runs on Logistic Regression only. RF and GBT use fixed Phase 2 hyperparameters due to the 1 GB Spark Connect ML cache cap. Details in reports/Phase3_summary.pdf .

Phase 3 Files Added
data/processed/ny_flights_phase3_bronze.csv -Databricks input
notebooks/databricks/ ← 6 exported notebooks
Phase 3 Data Sources

BTS On-Time Performance (same as Phase 1/2) — manual download
Meteostat Hourly Weather (new) — fetched programmatically inside Notebook 05 via the meteostat Python library

Only ny_flights_phase3_bronze.csv needs to be uploaded to Databricks. Weather is pulled live from Meteostat.


