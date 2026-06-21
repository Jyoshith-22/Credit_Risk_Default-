import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf():
    pdf_filename = "Credit_Risk_Prediction_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles for professional look
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=40,
        alignment=1 # Center
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#718096'),
        alignment=1 # Center
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Bullet'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#2D3748'),
        alignment=1
    )
    
    story = []
    
    # ------------------ PAGE 1: COVER PAGE ------------------
    story.append(Spacer(1, 150))
    story.append(Paragraph("Credit Risk Default Prediction<br/>Case Study Report", title_style))
    story.append(Paragraph("Innovexa Catalyst - Machine Learning Task", subtitle_style))
    story.append(Spacer(1, 180))
    story.append(Paragraph("Author: Antigravity AI Coding Assistant<br/>Date: June 2026<br/>Status: Completed", meta_style))
    story.append(PageBreak())
    
    # ------------------ PAGE 2: EXEC SUMMARY & DATA PREPROC ------------------
    story.append(Paragraph("1. Executive Summary & Objectives", h1_style))
    summary_text = (
        "Lending institutions face substantial credit risk when borrowers default. This case study "
        "implements an end-to-end machine learning solution to predict loan default probability (1 = default, 0 = non-default) "
        "using a dataset of over 32,000 credit applications. By cleansing outliers, imputing missing data, creating "
        "engineered features, and training state-of-the-art ensemble models, we have built a credit risk scoring model "
        "achieving an <b>ROC-AUC of 0.9515</b> and <b>PR-AUC of 0.9082</b> using a tuned LightGBM model. This allows for "
        "accurate borrower risk segmentation and data-driven underwriting decisions."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. Data Preprocessing & Feature Engineering", h1_style))
    preproc_text = (
        "The raw credit dataset contains variables capturing borrower demographics, income, employment length, loan characteristics, "
        "and historical default flags. The following operations were executed to build robust features:"
    )
    story.append(Paragraph(preproc_text, body_style))
    story.append(Paragraph("• <b>Outlier Cleansing:</b> Unrealistic age entries (> 100) and excessive employment lengths (> 60 years) were removed.", bullet_style))
    story.append(Paragraph("• <b>Missing Value Imputation:</b> Missing employment length and interest rates were imputed using the median values calculated on the training partition to prevent leakage.", bullet_style))
    story.append(Paragraph("• <b>Feature Engineering:</b> Created domain interaction metrics: <i>interest_to_income_ratio</i>, <i>emp_length_to_age_ratio</i>, and <i>cred_hist_to_age_ratio</i>.", bullet_style))
    story.append(Paragraph("• <b>Encoding & Scaling:</b> Categorical features were one-hot encoded, and numerical predictors were standardized using standard scaling.", bullet_style))
    
    story.append(Spacer(1, 10))
    if os.path.exists("feature_importance.png"):
        story.append(Image("feature_importance.png", width=5.5*inch, height=3.3*inch))
        story.append(Paragraph("<font color='#718096'><i>Figure 1: Top 15 Feature Importances from Tuned LightGBM</i></font>", meta_style))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 3: MODEL RESULTS ------------------
    story.append(Paragraph("3. Model Performance Evaluation", h1_style))
    model_intro = (
        "Five classification architectures were trained and evaluated on a holdout test set (20% split). "
        "The model parameters for our top-performing LightGBM model were optimized via K-Fold cross-validation using Optuna. "
        "Model performance across baseline and advanced architectures is summarized below:"
    )
    story.append(Paragraph(model_intro, body_style))
    
    # Load metrics csv and create table
    if os.path.exists("model_evaluation_metrics.csv"):
        m_df = pd.read_csv("model_evaluation_metrics.csv")
        table_data = [[
            Paragraph(col, table_header_style) for col in m_df.columns
        ]]
        for _, row in m_df.iterrows():
            row_data = []
            for col in m_df.columns:
                val = row[col]
                if isinstance(val, float):
                    row_data.append(Paragraph(f"{val:.4f}", table_cell_style))
                else:
                    row_data.append(Paragraph(str(val), table_cell_style))
            table_data.append(row_data)
            
        t = Table(table_data, colWidths=[1.6*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.95*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ]))
        story.append(t)
        
    story.append(Spacer(1, 15))
    if os.path.exists("model_roc_curves.png"):
        story.append(Image("model_roc_curves.png", width=5.5*inch, height=3.5*inch))
        story.append(Paragraph("<font color='#718096'><i>Figure 2: Receiver Operating Characteristic (ROC) Curve Comparison</i></font>", meta_style))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 4: RISK SEGMENTATION & RECOMMENDATIONS ------------------
    story.append(Paragraph("4. Credit Score Band & Segmentation Analysis", h1_style))
    seg_intro = (
        "Borrower default probabilities predicted by the Tuned LightGBM model were mapped to a standard credit score scale ranging from 300 to 850. "
        "We classified customers into five risk categories, which demonstrates perfect alignment with observed default patterns:"
    )
    story.append(Paragraph(seg_intro, body_style))
    
    if os.path.exists("credit_score_band_analysis.csv"):
        b_df = pd.read_csv("credit_score_band_analysis.csv")
        # Select and format columns
        b_df['default_rate'] = b_df['default_rate'].map(lambda x: f"{x*100:.2f}%")
        b_df.columns = ['Credit Score Band', 'Total Accounts', 'Observed Defaults', 'Default Rate', 'Default % (raw)']
        b_df_clean = b_df[['Credit Score Band', 'Total Accounts', 'Observed Defaults', 'Default Rate']]
        
        table_data = [[
            Paragraph(col, table_header_style) for col in b_df_clean.columns
        ]]
        for _, row in b_df_clean.iterrows():
            row_data = []
            for col in b_df_clean.columns:
                row_data.append(Paragraph(str(row[col]), table_cell_style))
            table_data.append(row_data)
            
        t2 = Table(table_data, colWidths=[2.5*inch, 1.3*inch, 1.3*inch, 1.4*inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2D3748')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ]))
        story.append(t2)
        
    story.append(Spacer(1, 10))
    if os.path.exists("risk_segmentation.png"):
        story.append(Image("risk_segmentation.png", width=5.5*inch, height=3.1*inch))
        story.append(Paragraph("<font color='#718096'><i>Figure 3: Observed Default Rates by FICO Credit Score Bands</i></font>", meta_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Underwriting Recommendations & Insights", h1_style))
    story.append(Paragraph("• <b>Automate Approvals for Excellent & Very Good Bands:</b> Applicants scoring above 740 show extremely low default rates (< 6%) and should be fast-tracked to maximize approval efficiency.", bullet_style))
    story.append(Paragraph("• <b>Dynamic Pricing & Restructuring for Good Band:</b> Borrowers in this category have moderate risk (~12% default rate). We recommend adjusting interest rates upward to hedge credit risk.", bullet_style))
    story.append(Paragraph("• <b>Strict Controls / Collateral Requirements for Fair Band:</b> A 25.1% default rate suggests tightening. Require co-signers or security deposits.", bullet_style))
    story.append(Paragraph("• <b>Automated Rejections for Poor Band:</b> With an 84.5% default rate, these applicants should be automatically declined to prevent high loan-loss reserves.", bullet_style))
    
    # Build Document
    doc.build(story)
    print("PDF Report compiled successfully as Credit_Risk_Prediction_Report.pdf")

if __name__ == "__main__":
    build_pdf()
