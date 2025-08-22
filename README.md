# 🚗 Car Price Prediction in Vietnam  

A complete machine learning pipeline built to estimate the prices of new and used cars based on real-world data collected from major Vietnamese online car marketplaces.  

The workflow covers: **data collection → database storage → cleaning & preprocessing → exploratory data analysis (EDA) → model training → deployment via a Flask-based chatbot**.  

---

## 📌 Objectives & Applications  
- **Objective**: Predict a fair selling price for cars listed in Vietnam’s e-commerce platforms.  
- **Real-World Applications**:  
  - Help buyers and sellers estimate a reasonable market price.  
  - Assist dealerships and marketplaces in detecting overpriced or underpriced listings.  

---

## 📊 Data & Technologies  
- **Data Sources**:  
  - oto.com.vn (~1,490 listings)  
  - bonbanh.com (~2,906 listings)  
  - xe.chotot.com (~9,113 listings)  
- **Total Records (after cleaning)**: 11,194 cars  
- **Number of Features**: 10  
  - `name`, `price`, `year`, `body_type`, `status`, `origin`, `mileage`, `fuel_type`, `brand`, `age`  

**Technologies Used**:  
- Python, pandas, NumPy, scikit-learn  
- BeautifulSoup, requests, aiohttp, async (web scraping)  
- PostgreSQL (data storage)  
- Power BI (data analysis & visualization)  
- Flask (chatbot deployment)  

---

## 🔧 Data Processing Pipeline  
1. **Data Collection**  
   - Web scraping from 3 major marketplaces.  
   - Stored raw data in PostgreSQL for easy reuse.  

2. **Cleaning & Preprocessing**  
   - Removed records with missing prices or invalid years.  
   - Filtered outliers using statistical methods.  
   - Encoded categorical variables with OneHotEncoder.  
   - Engineered new feature: vehicle age.  
   - Saved the cleaned dataset back to the database.  

3. **Exploratory Data Analysis (EDA)**  
   - Built interactive dashboards in **Power BI**.  
   - Used Seaborn/Matplotlib for correlation heatmaps and trend analysis.  
   - Explored price distributions by brand, mileage, age, and body type.  

---

## 🤖 Model Training & Evaluation  

**Baseline Models**:  

| Model             | R² Score | RMSE (VND)        |
|-------------------|----------|------------------|
| Linear Regression | 0.6076   | 161,542,607.54   |
| Ridge Regression  | 0.6099   | 161,055,913.93   |
| Lasso Regression  | 0.6076   | 161,545,421.98   |
| Random Forest     | 0.7712   | 123,335,749.67   |

**After Hyperparameter Tuning**:  
- **Train R²**: 0.9009  
- **Test R²**: 0.7721  
- **5-Fold Cross-Validation Mean R²**: 0.7345  

👉 **Random Forest** achieved the best performance.  

---

## 🖥️ Deployment  
- Built a lightweight chatbot with Flask.  
- Users enter car details (brand, year, mileage, condition, etc.) and receive a predicted price.  
- Frontend ready for API integration or scaling.  

---

## 📌 Reporting & Visualization  
- Power BI Dashboard: Cars-Insights.pbix
- PDF Report: Cars-Insights.pdf

---

## 📁 Project Structure  
```plaintext
CAR_PRICE_PREDICTION/
├── app/                         # Chatbot application
│   ├── templates/               # Frontend UI
│   │   └── index.html
│   └── app_car.py               # Flask app
│
├── data/                        # Raw and cleaned data
│   ├── cars_data.csv
│   ├── cars_data2.csv
│   └── data_cleaned.csv
│
├── model/                       # Trained models and encoders
│   ├── random_forest_model_1.joblib
│   ├── onehot_encoder.pkl
│   └── scaler.pkl
│
├── notebooks/                   # Analysis & training notebooks
│   ├── clean_data.ipynb
│   ├── crawl_data.ipynb
│   ├── train_model.ipynb
│   └── visualization.ipynb
│
├── power-bi/                    # Visualization reports
│   ├── Cars-Insights.pbix   # Power BI Dashboard
│   └── report/
│       └── Cars-Insights.pdf
│       └── Cars-Insights.pbix   # Power BI Dashboard
│
├── config.py                    # App & Database configuration
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Runtime version
├── README.md                    # Project documentation
└── .env                         # Database credentials (gitignored)
