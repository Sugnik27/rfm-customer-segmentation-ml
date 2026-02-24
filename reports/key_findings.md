# Key Findings — E-commerce Customer Segmentation and Prediction

---

## 📁 Dataset Overview
- **Source:** UCI Online Retail Dataset
- **Total Rows:** 541,909
- **Total Columns:** 8
- **Columns:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## 🔍 Data Loading Summary (01_data_loading.ipynb)

| Issue | Count | Action |
|---|---|---|
| Total Rows | 541,909 | — |
| Total Columns | 8 | — |
| Missing CustomerIDs | 135,080 (24.9%) | Drop — no use without CustomerID for RFM |
| Missing Descriptions | 1,454 | Drop |
| Cancelled Orders | 9,288 | Drop — marked with 'C' prefix in InvoiceNo |
| Negative Quantity Rows | 10,624 | Drop — returns/cancellations |
| Zero/Negative Price Rows | 2,517 | Drop — invalid transactions |
| Duplicate Rows | 5,268 | Drop |

### Key Observation:
- Missing CustomerIDs is the biggest data quality issue at ~25% of total data
- After cleaning, expected rows remaining: ~380,000 - 400,000
- CustomerID stored as **string** — because integer columns cannot hold NaN values in pandas, and CustomerID is an identifier not a number. String CustomerIDs also group cleanly without float formatting issues during RFM aggregation.

---

## 🧹 Data Cleaning Summary (02_data_cleaning.ipynb)
*(To be updated after cleaning)*

---

## 📊 EDA Key Findings (03_eda.ipynb)
*(To be updated after EDA)*

---

## ⚙️ RFM Feature Engineering (04_rfm_engineering.ipynb)
*(To be updated after RFM)*

---

## 🔵 Segmentation Results (05_segmentation.ipynb)
*(To be updated after clustering)*

---

## 🟢 Prediction Results (06_prediction.ipynb)
*(To be updated after supervised ML)*

---

## 💡 Business Recommendations
*(To be updated after segmentation)*

---

*Updated progressively throughout the project*
