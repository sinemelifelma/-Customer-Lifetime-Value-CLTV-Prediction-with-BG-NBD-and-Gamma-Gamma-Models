# 🧮 Customer Lifetime Value (CLTV) Prediction with BG-NBD and Gamma-Gamma Models

## 🎯 Business Problem
FLO wants to estimate the potential future value of its existing customers to guide its sales and marketing strategies.

## 📦 Dataset
The dataset includes omnichannel customer data from 2020–2021, containing both online and offline purchase behaviors.

**Main features:**
- master_id
- first_order_date, last_order_date
- total orders and total spending (online & offline)
- interested_in_categories_12

## 🧹 Data Preparation
- Handled outliers using the IQR method  
- Converted date columns to datetime  
- Created `total_order_number` and `total_customer_value` features  

## 📈 Modeling
1. **BG/NBD model** → predicts future purchase frequency  
2. **Gamma-Gamma model** → predicts average transaction value  
3. Combined for **6-month CLTV estimation**

## 🧩 Segmentation
Customers were divided into 4 CLTV-based segments: A (Top) → D (Lowest).

## 📊 Insights
Segment A customers have the highest frequency and monetary averages, representing the company’s most valuable group.

---

## 🔗 Links
- 📖 [Medium Article](https://medium.com/@sinemelifelma/customer-lifetime-value-cltv-prediction-with-bg-nbd-and-gamma-gamma-models-bd52daa99cd5)
- 💾 [Kaggle Notebook](https://www.kaggle.com/code/sinemelifelma/cltv-prediction-with-bg-nbd-and-gamma-gamma)
- 💼 [LinkedIn Post](https://www.linkedin.com/in/sinem-elif-elma-bab7579b/)


## 👩‍💻 Author
**Sinem Elif Elma**  
Data Analyst | Data Scientist | Lifelong Learner  
📍 Netherlands  

