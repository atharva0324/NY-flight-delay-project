# Flight Delay Analysis – Phase 1

## 📌 Project Overview

This project analyzes flight delays for flights departing from New York State airports.
The goal is to explore patterns in arrival delays (ArrDel15) using exploratory data analysis (EDA).

The dataset includes monthly flight records from December 2024 to October 2025 obtained from the Bureau of Transportation Statistics (BTS).

## Data Source

This project uses the U.S. DOT Bureau of Transportation Statistics
On-Time Performance dataset (1987–Present)
Following are the link to this datasets:-

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


---

## 📓 Original Exploratory Notebook

The initial exploratory analysis and early cleaning steps were developed in a Jupyter Notebook.

You can view it here:

notebooks/flight_project_exploration.ipynb

Note: The final reproducible pipeline has been modularized into Python scripts inside the `src/` directory.