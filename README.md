# AutoPulse PK 🚗 - Used Car Fair-Value & Deal Analyzer

AutoPulse PK is a full-stack machine learning application designed to predict the fair market value of used cars in Pakistan. By leveraging historical listings (PakWheels, OLX), the system helps buyers and sellers bypass market volatility and information asymmetry to make data-driven pricing decisions.

This project was developed as a final academic submission for the University AI Lab.

---

## 🌟 Key Features

* **Dual-Model Inference:** Uses a combination of Regression (exact PKR price estimation) and Classification (market segmentation: Budget, Mid-range, Premium).
* **Multi-Model Support:** Features three distinct ML pipelines built by team members, accessible via a dynamic model switcher in the UI.
* **Deal Verdicts:** Compares a user's asking price against the model's prediction to flag overpriced cars or hidden deals.
* **Automated PDF Reporting:** Includes a built-in Python script using `fpdf2` to dynamically generate a comprehensive academic project report from the latest evaluation metrics and charts.

---

## 🧠 Machine Learning Architecture

The system evaluates cars using a robust 13-feature pipeline trained with `StratifiedGroupKFold` to prevent relisting data leakage. 

**Team Model Implementations:**
* **Faiz:** Random Forest (Regression) & Decision Tree (Classification)
* **Ahmed:** Ridge Regression & K-Nearest Neighbors (Classification)
* **Farhan:** Lasso Regression & Decision Tree (Classification)

---

## ⚠️ Prerequisites: Download the Models

To comply with GitHub's 100 MB file size limit, the trained `.joblib` models are hosted externally on Google Drive. **You must download these models before running the backend.**

1. Go to our [Google Drive Model Folder](https://drive.google.com/drive/folders/1cL58391ZRsGwYp0_4suTLYAG4knbbDCg?usp=sharing).
2. Download all `.joblib` files.
3. Place the downloaded files directly into the `models/` directory at the root of this project (create the folder if it does not exist).

---

## 🚀 Installation & Setup

Ensure you have **Python 3.10+** and **Node.js** installed on your system.

### 1. Clone the Repository
`git clone https://github.com/Faiz-Ul-Hassan-Khalil/autopulse.git`
`cd autopulse`

### 2. Start the FastAPI Backend
Open a terminal in the root directory:

`python -m venv .venv`
`source .venv/bin/activate  # On Windows use: .venv\Scripts\activate`

`pip install -r requirements.txt`

`cd backend`
`uvicorn main:app --reload`

*The backend API will be available at http://127.0.0.1:8000*

### 3. Start the React Frontend
Open a **new** terminal in the root directory:

`cd frontend`
`npm install`
`npm run dev`

*The UI will be available at http://localhost:5173*

---

## 📄 Generating the Academic Report

This repository includes a script to dynamically generate the final PDF project report based on the latest saved metrics and evaluation figures.

Run this command from the project root:
`python src/reports/generate_report.py`

This will output `AutoPulse_Final_Project_Report.pdf` directly into the root directory, perfectly formatted for submission.

---

## 📂 Project Structure

autopulse/
├── backend/                  # FastAPI server and ML inference endpoints
├── frontend/                 # React UI + Tailwind CSS dashboard
├── models/                   # (.joblib files - MUST BE DOWNLOADED FROM GOOGLE DRIVE)
├── notebooks/                # Jupyter notebooks for EDA and initial training
├── reports/                  
│   ├── evidence/             # CSV metrics, logs, and confusion matrices
│   └── figures/              # EDA and feature importance PNG charts
├── src/                      
│   ├── data/                 # Data cleaning and auditing scripts
│   ├── models/               # Model training pipelines for each team member
│   └── reports/              # fpdf2 script for automated PDF generation
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
