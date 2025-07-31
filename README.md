# 🚗 Used Car Price Prediction in Vietnam  

A complete machine learning pipeline to estimate the price of used (and new) cars based on real-world data scraped from popular Vietnamese online car marketplaces.  

This project covers the entire workflow: **data collection → storage in Database → cleaning → exploratory analysis → model training → deployment via a Flask-based chatbot**.  

---

## 📌 Project Overview  
**Objective:**  
- Predict a fair selling price for cars listed on Vietnamese e-commerce platforms.  

**Real-World Applications:**  
- Help users estimate a reasonable price before buying or selling a car.  
- Assist dealerships and platforms in detecting overpriced or underpriced listings.  

**Data Sources:**  
- [oto.com.vn](https://oto.com.vn/mua-ban-xe) — ~1,490 listings  
- [bonbanh.com](https://bonbanh.com/oto) — ~2,906 listings  

**Technologies Used:**  
- Python, pandas, scikit-learn, seaborn, matplotlib  
- BeautifulSoup (web scraping)  
- PostgreSQL (Database for storage)  
- Power BI (EDA & reporting)  
- Flask (chatbot deployment)  
- ChatGPT (pipeline building assistance)  

---

## 📊 Data Summary  
- **Total records after cleaning:** 3,592 cars  
- **Number of features:** 10  

| Column     | Data Type | Description |
|------------|-----------|-------------|
| name       | object    | Car name |
| price      | int64     | Price (VND) |
| year       | int32     | Year of manufacture |
| body_type  | object    | Body type |
| status     | object    | Condition (new/used) |
| origin     | object    | Origin |
| mileage    | float64   | Mileage (km) |
| fuel_type  | object    | Fuel type |
| brand      | object    | Brand |
| age        | int32     | Vehicle age (2025 - year) |

---

## 🔧 Data Processing Pipeline  

### 1. Data Collection  
- Web scraping from the two marketplaces.  
- Data stored in a **PostgreSQL Database** for easy reuse.  

### 2. Data Cleaning & Preprocessing  
- Retrieve data from the **Database**.  
- Remove records with missing price, year, or mileage.  
- Drop duplicates and outliers using statistical methods.  
- Apply **log transformation** to reduce skewness in numeric fields.  
- Encode categorical variables using OneHotEncoder.  
- Create a new feature: **vehicle age**.  
- Store the cleaned dataset back into the **Database**.  

### 3. Exploratory Data Analysis (EDA)  
- Combined **Power BI** dashboards with **Seaborn/Matplotlib** visualizations.  
- Analyzed pricing trends by brand, age, mileage, etc.  
- Used heatmaps to explore feature correlations.  

---

## 🤖 Model Training & Evaluation  

A comparison of regression models:  

| Model             | R² Score | RMSE (VND)        |
|-------------------|----------|------------------|
| Linear Regression | 0.6756   | 183,166,563.06   |
| Ridge Regression  | 0.6842   | 180,733,663.38   |
| Lasso Regression  | 0.6763   | 182,962,184.19   |
| Random Forest     | 0.8127   | 139,168,949.55   |

**After hyperparameter tuning:**  
- Train R²: 0.9340  
- Test R²: 0.8168  
- Cross-Validation (5-fold) Mean R²: 0.8014  

👉 **Random Forest** delivered the best performance.  

---

## 🖥️ Deployment  
- Built a lightweight chatbot with Flask.  
- Users enter car details (brand, year, mileage, condition, etc.) and receive a predicted price.  
- Frontend ready for API integration or scaling.  

---

## ⚙️ How to Run  

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app/app_car.py


## 📁 Project Structure

```
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
│   └── report/
│       ├── Cars-Insights.pdf
│       └── Cars-Insights.pbix
│
├── config.py                    # App & Database configuration
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Runtime version
├── README.md                    # Project documentation
└── .env                         # Database credentials (gitignored)

```


## 📌 Reporting & Visualization  
- Power BI Dashboard: **Cars-Insights.pbix**  
- PDF Report: **Cars-Insights.pdf**  

