# E-commerce Customer Segmentation & Prediction

A machine learning project that segments e-commerce customers using 
RFM Analysis and predicts customer segments for new customers.

Live App link: https://rfm-customer-segmentation-ml.streamlit.app/

## Project Structure
```
E-commerce_customer_segementation_and_prediction/
├── data/
│   ├── cleaned_data.csv
│   ├── rfm_data.csv
│   ├── rfm_train_scaled.csv
│   └── rfm_test_scaled.csv
├── notebooks/
│   ├── 01_metadata.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_rfm_engineering.ipynb
│   ├── 05_segmentation.ipynb
│   └── 06_prediction.ipynb
├── src/
│   ├── preprocessor.py
│   ├── training.py
│   ├── deployment.py
│   └── app.py
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## Customer Segments
- 🏆 Champions — Recent, frequent, high spenders
- 💛 Loyal Customers — Consistent buyers
- ⚠️ At-Risk Customers — Declining engagement
- 💤 Lost Customers — Inactive customers

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Sugnik27/rfm-customer-segmentation-ml.git
cd rfm-customer-segmentation-ml
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run training pipeline
```bash
python src/training.py
```

### 5. Run Streamlit app
```bash
streamlit run src/app.py
```

## Tech Stack
- Python 3.13
- Scikit-learn
- XGBoost
- Streamlit
- Pandas
- NumPy

## Dataset
UCI Online Retail Dataset — 541,909 transactions, 
4,338 customers, December 2010 to December 2011.

## Model Performance
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.9965 | 0.9965 |
| Random Forest | 0.9862 | 0.9861 |
| XGBoost | 0.9827 | 0.9827 |

## Author
Sugnik Mondal — Boston Institute of Analytics, Manipal University Jaipur
```
