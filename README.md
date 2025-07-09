# 🚗 Used Car Price Prediction (Vietnam)

An end-to-end machine learning pipeline to estimate used car prices based on real-world data from Vietnamese marketplaces.  
This project includes data scraping, cleaning, exploratory data analysis (EDA), model training, and deployment via a chatbot interface using Flask.

## 📌 Project Overview

- **Goal**: Predict the selling price of used cars listed on Vietnamese car trading platforms.
- **Data Source**: Publicly available listings from [https://oto.com.vn/](https://oto.com.vn/)  
- **Tech Stack**: Python, pandas, scikit-learn, Seaborn, BeautifulSoup, Flask, ChatGPT (for assisted coding)

---

## 📊 Dataset

- **Raw listings scraped**: ~1,495  
- **Final dataset after cleaning**: 1,343 rows, 13 features  
- **Key features**:
  - `brand`, `model`, `body_type`, `year`, `mileage`, `engine`, `location`, etc.

---

## 🔧 Pipeline Stages

### 1. Data Collection
- Used Python with `BeautifulSoup` and `requests` to scrape used car listings.
- Stored raw data as CSV for reproducibility.

### 2. Data Cleaning & Preprocessing
- Removed duplicates and irrelevant rows.
- Handled missing values.
- Applied log transformation on `price` and `mileage`.
- Removed extreme outliers.
- Encoded categorical variables.

### 3. Exploratory Data Analysis (EDA)
- Visualized price distributions by brand and year.
- Correlation heatmaps to evaluate feature relationships.

### 4. Modeling
Trained and evaluated multiple regression models:
| Model              | R² Score | MSE         |
|-------------------|----------|-------------|
| Linear Regression | 0.7403   | 3.79e+16    |
| Ridge Regression  | 0.7458   | 3.71e+16    |
| Lasso Regression  | 0.7403   | 3.79e+16    |
| **Random Forest** | **0.8027** | **2.88e+16** |

