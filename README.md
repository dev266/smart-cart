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

Pair plots were used to inspect relationships and possible outliers.

Outliers were removed using the following conditions:

```python
df_cleaned = df_cleaned[df_cleaned["Age"] < 90]

df_cleaned = df_cleaned[
    df_cleaned["Income"] < 600_000
]
```

## 5. Correlation Analysis

A correlation matrix was created for numerical features.

```python
corr = df_cleaned.corr(numeric_only=True)
```

A heatmap was then used to visualize relationships between customer features.

## 6. Encoding Categorical Features

Categorical columns were converted into numerical values using `OneHotEncoder`.

The encoded columns include:

```text
Education
Living_With
```

Example:

```python
ohe = OneHotEncoder()

enc_cols = ohe.fit_transform(
    df_cleaned[cat_cols]
)
```

## 7. Feature Scaling

Before applying PCA and clustering, the features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

## 8. Principal Component Analysis

PCA was used to reduce the dataset into three dimensions.

```python
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)
```

The reduced data was visualized using a 3D scatter plot.

This makes it easier to visually inspect the structure of the customer groups.

## 9. Choosing the Number of Clusters

Two techniques were used to analyze the optimal number of clusters.

### Elbow Method

K-Means was tested for values of `K` from 1 to 10.

```python
for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )
```

WCSS was recorded for each value of K.

`KneeLocator` was also used to identify the elbow point automatically.

### Silhouette Score

Silhouette scores were calculated for K values from 2 to 10.

```python
score = silhouette_score(
    X_pca,
    labels
)
```

Both WCSS and Silhouette Score were visualized to help select an appropriate number of clusters.

## 10. K-Means Clustering

K-Means clustering was applied using:

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

labels_kmeans = kmeans.fit_predict(X_pca)
```

The four clusters were visualized in 3D PCA space.

## 11. Agglomerative Clustering

Agglomerative Clustering was also applied.

```python
agg_clf = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

labels_agg = agg_clf.fit_predict(X_pca)
```

This provided another clustering approach for comparison.

## 12. Cluster Characterization

The Agglomerative Clustering labels were added back to the dataset:

```python
X["cluster"] = labels_agg
```

The distribution of customers across clusters was visualized using a count plot.

## Income and Spending Analysis

A scatter plot was used to analyze the relationship between:

```text
Total Spending
Income
Customer Cluster
```

This helps identify different customer profiles based on financial behaviour.

## Cluster Summary

Average feature values were calculated for each cluster:

```python
cluster_summary = X.groupby(
    "cluster"
).mean()
```

This summary can be used to understand the characteristics of each customer group.

## Clustering Methods Used

| Method | Purpose |
|---|---|
| PCA | Dimensionality reduction |
| Elbow Method | Estimate optimal K |
| Silhouette Score | Evaluate cluster quality |
| K-Means | Partition customers into groups |
| Agglomerative Clustering | Hierarchical customer segmentation |

## Key Concepts Practiced

- Data preprocessing
- Missing value handling
- Feature engineering
- Outlier removal
- Exploratory Data Analysis
- Correlation analysis
- One-Hot Encoding
- Feature scaling
- PCA
- K-Means clustering
- Agglomerative clustering
- Elbow Method
- Silhouette Score
- Cluster characterization
- Customer segmentation

## Project Structure

```text
customer-segmentation/
│
├── smartcart_customers.csv
├── customer_segmentation.ipynb
├── README.md
└── requirements.txt
```

## Installation

Install the required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn kneed jupyter
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

## Future Improvements

- Compare additional clustering algorithms such as DBSCAN
- Perform more detailed customer persona analysis
- Analyze cluster-specific purchasing behaviour
- Use additional dimensionality reduction methods
- Evaluate clusters using additional metrics
- Build an interactive customer segmentation dashboard
- Use clusters for targeted marketing recommendations