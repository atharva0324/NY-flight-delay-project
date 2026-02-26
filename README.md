# Flight Delay Analysis – Phase 1

## 📌 Project Overview

This project analyzes flight delays for flights departing from New York State airports.
The goal is to explore patterns in arrival delays (ArrDel15) using exploratory data analysis (EDA).

The dataset includes monthly flight records from December 2024 to October 2025 obtained from the Bureau of Transportation Statistics (BTS).

## Data Requirement

Download monthly BTS On-Time Performance CSV files and place them inside:

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