from fpdf import FPDF
import pandas as pd
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT_DIR / "reports" / "evidence"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
OUTPUT_PDF = ROOT_DIR / "AutoPulse_Final_Project_Report.pdf"

class ProjectReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "AutoPulse PK: Used Car Fair-Value & Deal Analyzer", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "I", 11)
        self.cell(0, 8, "Final AI Lab Project Report", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 28, 200, 28)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.ln(5)
        self.set_font("helvetica", "B", 14)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def body_text(self, text):
        self.set_font("helvetica", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_image_safe(self, img_path, width=150, title=None):
        if Path(img_path).exists():
            if title:
                self.set_font("helvetica", "B", 10)
                self.cell(0, 6, title, align="C", new_x="LMARGIN", new_y="NEXT")
            # Center the image
            x_pos = (210 - width) / 2
            self.image(str(img_path), x=x_pos, w=width)
            self.ln(5)

def create_full_report():
    pdf = ProjectReportPDF()
    pdf.add_page()
    
    # --- 1. ABSTRACT & PROBLEM STATEMENT ---
    pdf.section_title("1. Problem Statement & Project Overview")
    pdf.body_text(
        "The Pakistani used car market suffers from high volatility and information asymmetry. Buyers and sellers "
        "rely heavily on manual negotiation, resulting in inconsistent pricing. AutoPulse PK solves this by providing "
        "a data-driven, machine learning pipeline that estimates the fair market value of a vehicle based on historical "
        "PakWheels and OLX data. It classifies listings into Budget, Mid-range, and Premium segments, and flags "
        "potential hidden deals or overpriced listings."
    )

    # --- 2. DATA PREPROCESSING & CLEANING ---
    pdf.section_title("2. Data Preprocessing & Cleaning")
    cleaning_log = EVIDENCE_DIR / "faiz_02_cleaning_log.txt"
    if cleaning_log.exists():
        with open(cleaning_log, 'r', encoding='utf-8') as f:
            pdf.body_text(f.read())
    else:
        pdf.body_text(
            "Data was cleaned by deduplicating market listings, handling missing values in Japanese Auction Grades, "
            "and normalizing engine displacement (cc) and mileage (km). We utilized StratifiedGroupKFold to prevent "
            "data leakage from relisted vehicles."
        )

    # --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
    pdf.add_page()
    pdf.section_title("3. Exploratory Data Analysis (EDA)")
    eda_insights = EVIDENCE_DIR / "faiz_03_eda_insights.txt"
    if eda_insights.exists():
        with open(eda_insights, 'r', encoding='utf-8') as f:
            pdf.body_text(f.read())
            
    # Add EDA Figures dynamically
    pdf.add_image_safe(FIGURES_DIR / "fig1_price_distribution_skewness.png", width=130, title="Price Distribution & Skewness")
    pdf.add_image_safe(FIGURES_DIR / "fig4_correlation_heatmap.png", width=140, title="Feature Correlation Heatmap")

    # --- 4. METHODOLOGY & PIPELINE ---
    pdf.add_page()
    pdf.section_title("4. Methodology & Machine Learning Pipeline")
    pdf.body_text(
        "The project utilizes a dual-model architecture per team member, consisting of:\n"
        "1. A Regression Model: To predict the exact continuous PKR price of the vehicle.\n"
        "2. A Classification Model: To categorize the vehicle into discrete market segments (Budget, Mid-range, Premium).\n\n"
        "Team members developed independent pipelines to compare different algorithmic approaches:"
    )
    pdf.body_text("- Faiz: Random Forest (Regression) & Decision Tree (Classification)")
    pdf.body_text("- Ahmed: Ridge Regression & K-Nearest Neighbors (Classification)")
    pdf.body_text("- Farhan: Lasso Regression & Decision Tree (Classification)")

    # --- 5. RESULTS & MODEL COMPARISON ---
    pdf.add_page()
    pdf.section_title("5. Model Evaluation & Comparison")
    
    team_members = ["faiz", "ahmed", "farhan"]
    for member in team_members:
        metrics_file = EVIDENCE_DIR / f"{member}_metrics.csv"
        if metrics_file.exists():
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 8, f"{member.capitalize()}'s Pipeline Results", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            
            df = pd.read_csv(metrics_file)
            for index, row in df.iterrows():
                if row.get('task') == 'regression' and row.get('set') == 'test':
                    pdf.cell(0, 6, f"  - Regression (Test): R² = {row['R2']:.3f} | MAE = PKR {row['MAE']:,.0f}", new_x="LMARGIN", new_y="NEXT")
                elif row.get('task') == 'classification' and row.get('set') == 'test':
                    acc = row.get('Accuracy', row.get('accuracy', 0))
                    pdf.cell(0, 6, f"  - Classification (Test): Accuracy = {acc*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # --- 6. FEATURE IMPORTANCE & CONFUSION MATRICES ---
    pdf.section_title("6. Feature Importance & Confusion Matrices")
    pdf.body_text("The following visuals detail the primary drivers of price and the classification accuracy across price bands.")
    
    pdf.add_image_safe(EVIDENCE_DIR / "faiz_rf_feature_importance.png", width=140, title="Faiz - Random Forest Feature Importance")
    pdf.add_image_safe(EVIDENCE_DIR / "ahmed_knn_clf_confusion_matrix.png", width=120, title="Ahmed - KNN Confusion Matrix")

    # --- 7. SYSTEM ARCHITECTURE & UI ---
    pdf.add_page()
    pdf.section_title("7. Application Deployment & User Interface")
    pdf.body_text(
        "To make the models accessible, a full-stack web application was developed:\n\n"
        "- Backend (FastAPI): Exposes endpoints to load the selected team member's .joblib models dynamically and run inference.\n"
        "- Frontend (React + Tailwind CSS): Provides a dashboard interface allowing users to input car specifications, compare asking prices against model predictions, and evaluate whether a listing is a good deal."
    )

    pdf.section_title("8. Conclusion")
    pdf.body_text(
        "The AutoPulse PK system successfully demonstrates that machine learning can accurately model the complex pricing "
        "dynamics of the Pakistani used car market. The integration of regression for precise pricing and classification for "
        "segmentation provides a robust, multi-faceted tool for buyers and sellers."
    )

    pdf.output(str(OUTPUT_PDF))
    print(f"Comprehensive report successfully generated at: {OUTPUT_PDF}")

if __name__ == "__main__":
    create_full_report()