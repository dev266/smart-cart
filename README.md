# Customer Segmentation using Machine Learning

A machine learning project that segments customers into meaningful groups based on demographics, income, spending behaviour, household information, and customer activity.

The project applies data preprocessing, feature engineering, dimensionality reduction, and unsupervised learning techniques to identify distinct customer groups.

## Project Overview

Customer segmentation helps businesses understand different types of customers and tailor marketing strategies accordingly.

This project performs customer segmentation using:

- K-Means Clustering
- Agglomerative Clustering
- Principal Component Analysis (PCA)
- Elbow Method
- Silhouette Score

The final clustering analysis uses **4 customer clusters**.

## Dataset

The dataset is loaded from:

```text
smartcart_customers.csv
```

It includes customer information such as:

- Income
- Year of Birth
- Education
- Marital Status
- Customer Joining Date
- Recency
- Number of Children
- Product Spending
- Campaign Response
- Other customer behaviour features

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Kneed
- Jupyter Notebook

## Project Workflow

### 1. Data Preprocessing

The dataset was first inspected using:

```python
df.head()
df.shape
df.isnull().sum()
```

Missing values in the `Income` column were handled using median imputation:

```python
df["Income"] = df["Income"].fillna(df["Income"].median())
```

## 2. Feature Engineering

Several new features were created to make the dataset more useful for clustering.

### Age

```python
df["Age"] = 2026 - df["Year_Birth"]
```

### Customer Tenure

Customer joining dates were converted into datetime format.

```python
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
```

Customer tenure was calculated using the latest joining date as a reference:

```python
reference_date = df["Dt_Customer"].max()

df["Customer_Tenure_Days"] = (
    reference_date - df["Dt_Customer"]
).dt.days
```

### Total Spending

A new `Total_Spending` feature was created by combining spending across different product categories:

```python
df["Total_Spending"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)
```

### Total Children

```python
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
```

### Education Groups

Education categories were simplified into:

```text
Undergraduate
Graduate
Postgraduate
```

### Living Status

Marital status was converted into two broader categories:

```text
Partner
Alone
```

## 3. Dropping Unnecessary Columns

Columns that were either identifiers or already represented by engineered features were removed.

Examples include:

- ID
- Year_Birth
- Marital_Status
- Kidhome
- Teenhome
- Dt_Customer
- Individual product spending columns

## 4. Outlier Detection and Removal

Pair plots were used to inspect relationships and