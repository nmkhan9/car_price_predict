# 🚗 Used Car Price Prediction in Vietnam

A complete **machine learning pipeline** to estimate the price of used cars based on real-world data scraped from popular Vietnamese e-commerce platforms for vehicles.

This project includes **data scraping, cleaning, exploratory analysis, model training, and deployment via a Flask-based chatbot**.

---

## 📌 Project Overview

**Objective**: Predict a fair selling price for used cars listed on online marketplaces in Vietnam.

**Real-World Applications**:
- Help users estimate a reasonable price before posting their vehicles.
- Assist car dealerships and platforms in detecting over/underpriced listings.

**Data Sources**:
- [oto.com.vn/mua-ban-xe](https://oto.com.vn/mua-ban-xe) — ~1,490 listings
- [bonbanh.com/oto](https://bonbanh.com/oto) — ~2,851 listings

**Technologies Used**:
- Python, pandas, scikit-learn, Seaborn, BeautifulSoup
- Flask (for chatbot deployment)
- ChatGPT-assisted pipeline building

---

## 📊 Data Summary

- **Total records after cleaning**: 3,557 cars  
- **Number of features**: 10  
- **Key features**: car name, price, year, condition, origin, mileage, fuel type, body type, brand, vehicle age

---

## 🔧 Data Processing Pipeline

### 1. Data Collection
- Web scraping was conducted on two vehicle trading platforms.
- Each record includes details like name, price, brand, mileage, fuel type, etc.
- Data was saved as `.csv` files for easy reuse.

### 2. Data Cleaning & Preprocessing
- Removed records with missing price, year, or mileage.
- Duplicates and outliers were dropped using statistical methods.
- Applied log transformation to reduce skewness on numeric fields.
- Encoded categorical variables like brand and fuel type.
- Created a new feature: **vehicle age** for better prediction accuracy.

### 3. Exploratory Data Analysis (EDA)
- Visualized relationships between price, mileage, year, and brand.
- Correlation heatmaps used to explore variable dependencies.

---

## 🤖 Model Training & Evaluation

A comparison of multiple regression models was conducted:

| Model              | R² Score | RMSE (VND)     |
|-------------------|----------|----------------|
| Linear Regression | 0.6779   | 189 million    |
| Ridge Regression  | 0.6809   | 188 million    |
| Lasso Regression  | 0.6784   | 189 million    |
| Random Forest     | **0.8255** | **139 million** |

**Random Forest** provided the best results thanks to its robustness to non-linear data and noise.

After hyperparameter tuning:
- **Train R² Score**: 0.9289  
- **Test R² Score**: 0.8320  
- **Cross-Validation (5-fold) R² Mean**: 0.7902

---

## 🖥️ Deployment

- Built a simple chatbot using Flask.
- Users input car information (brand, year, mileage, condition...) and receive a price estimate.
- Lightweight frontend, ready for API integration or scaling.

---

## ⚙️ How to Use

1. Install required libraries from `requirements.txt`
2. Run the Flask app locally.
3. Open your browser and visit `localhost` to start the chatbot.

---

## 📁 Project Structure
CAR_PRICE_PREDICTION/
├── data/                          # Raw and cleaned datasets
│   ├── cars_data.csv              # Raw data from oto.com.vn
│   ├── cars_data2.csv             # Raw data from bonbanh.com
│   ├── data_cleaned.csv           # Cleaned dataset used for modeling
│   └── Crawl_data.ipynb           # Web scraping notebook

├── notebooks/                     # Notebooks for analysis and modeling
│   ├── clean_data.ipynb           # Data cleaning & feature engineering
│   ├── train_model.ipynb          # Model training and evaluation
│   └── visualization.ipynb        # Exploratory Data Analysis (EDA)

├── model/                         # Trained models and encoders
│   ├── random_forest_model_1.joblib  # Trained Random Forest model
│   ├── onehot_encoder.pkl         # Encoder for categorical variables
│   ├── scaler.pkl                 # Scaler for numerical features
│   ├── app_car.py                     # Flask app to run the prediction chatbot
│   └── templates/
│       └── index.html             # Frontend for Flask chatbot app

├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
