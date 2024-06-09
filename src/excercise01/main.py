import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#1.  Verschaffen Sie sich einen Überblick über die Daten.

#csv file path
df = pd.read_csv('uncleaned2_bike_sales.csv')

#print(df.head())
#print(df.shape)
#print(df.columns)
#print(df.index)
#print(df.values)
#print(df.describe())
#print(df.info())

#2. Data Cleaning

## 2.1 kovertieren in Euro nach dollar

# Define the exchange rate from USD to Euro by 8.6.
usd_to_euro = 0.92 

# Convert prices from USD to Euro
df[' Unit_Cost_€'] = df[' Unit_Cost_$'] * usd_to_euro
df[' Unit_Price_€ '] = df[' Unit_Price_$ '] * usd_to_euro
df[' Profit_€ '] = df[' Profit_$ '] * usd_to_euro
df[' Cost_€'] = df[' Cost_$'] * usd_to_euro
df['Revenue_€'] = df['Revenue_$'] * usd_to_euro

#print(df.head())

#print(df.shape)
## 2.2  Löschen Sie Spalten (Features) in denen mehr als 60% der Einträge fehlen.

# Calculate threshold for columns and drop columns with more than 60% missing values
threshold_col = len(df) * 0.6 
df = df.dropna(thresh=threshold_col, axis=1)


## 2.3   Löschen Sie Zeilen (Records) in denen mehr als 60% der Einträge fehlen.

# Calculate threshold for rows and drop rows with more than 60% missing values
threshold_row = len(df.columns) * 0.6 
df = df.dropna(thresh=threshold_row, axis=0)

#print(df.shape)

## 2.4 Ergänzen Sie die restlichen fehlenden Einträge mit einer der folgenden Methoden
#missing_values = df.isnull().sum()
#print(missing_values)


#print(df.tail())
#print(df.columns)

# Fill missing values function
def fill_missing_values(df, column, method):
    if method == 'Random Replacement':
        print(f"Filling missing values in {column} using Random Replacement")
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            sample_values = df[column].dropna().sample(n=missing_count, replace=True).values
            df.loc[df[column].isnull(), column] = sample_values
    elif method == 'Fehlende Werte durch Querverweise ergänzen':
        print(f"Filling missing values in {column} using querverweise ergaenzen")
        if column == 'State':  # Only apply for 'State' column
            # Find the most common state for each country
            country_to_state = df.groupby('Country')['State'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()

            # Fill missing values based on country
            def fill_state_based_on_country(row):
                if pd.isnull(row['State']):
                    country = row['Country']
                    if country in country_to_state:
                        return country_to_state[country]
                return row['State']

            # Apply the function to fill missing values in 'State' column
            df['State'] = df.apply(fill_state_based_on_country, axis=1)
        elif column in ['Product_Category', 'Sub_Category']:  # Apply for 'Product_Category' and 'Sub_Category' columns
            # Find the most common sub-category for each product category
            product_category_to_sub_category = df.groupby('Product_Category')['Sub_Category'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()

            # Fill missing values based on product category
            def fill_sub_category_based_on_product_category(row):
                if pd.isnull(row['Sub_Category']):
                    product_category = row['Product_Category']
                    if product_category in product_category_to_sub_category:
                        return product_category_to_sub_category[product_category]
                return row['Sub_Category']

            # Apply the function to fill missing values in 'Sub_Category' column
            df[column] = df.apply(fill_sub_category_based_on_product_category, axis=1)

    elif method == 'Fill with zeros':
        print(f"Filling missing values in {column} using fill with zero")
        # Fill missing values with zeros
        df[column].fillna(0, inplace=True)
    elif method == 'Fill with mean':
        print(f"Filling missing values in {column} using fill with mean")
        df[column].fillna(df[column].mean(), inplace=True)  # Fill NaN values with mean
    elif method == 'stratified replacement':
        print(f"Filling missing values in {column} using stratified Replacement")
        pass

#print(df.isnull().sum())


fill_missing_values(df, 'Day', 'Random Replacement')
fill_missing_values(df, 'Customer_Age', 'Random Replacement')
fill_missing_values(df, 'Customer_Gender', 'Random Replacement')
fill_missing_values(df, 'State', 'Fehlende Werte durch Querverweise ergänzen')
fill_missing_values(df, 'Product_Catagory', 'Fehlende Werte durch Querverweise ergänzen')
fill_missing_values(df, 'Sub_Catagory', 'Fehlende Werte durch Querverweise ergänzen')
fill_missing_values(df, 'Order_Quantity', 'Random Replacement')
fill_missing_values(df, ' Unit_Cost_$', 'Fill with mean')
fill_missing_values(df, ' Unit_Price_$ ', 'Fill with mean')
fill_missing_values(df, ' Cost_$', 'Fill with mean')
fill_missing_values(df, 'Revenue_$', 'Fill with mean')
fill_missing_values(df, ' Profit_$ ', 'Fill with mean')

#print(df.tail())
#remove Nachkommastellen
df['Sales_Order #'] = df['Sales_Order #'].round(0).astype(int)
df['Day'] = df['Day'].round(0).astype(int)
df['Year'] = df['Year'].round(0).astype(int)
df['Customer_Age'] = df['Customer_Age'].round(0).astype(int)
df['Order_Quantity'] = df['Order_Quantity'].round(0).astype(int)


df[' Unit_Cost_$'] = df[' Unit_Cost_$'].round(2)
df[' Unit_Price_$ '] = df[' Unit_Price_$ '].round(2)
df[' Profit_$ '] = df[' Profit_$ '].round(2)
df[' Cost_$'] = df[' Cost_$'].round(2)
df['Revenue_$'] = df['Revenue_$'].round(2)
df[' Unit_Cost_€'] = df[' Unit_Cost_€'].round(2)
df[' Unit_Price_€ '] = df[' Unit_Price_€ '].round(2)
df[' Profit_€ '] = df[' Profit_€ '].round(2)
df[' Cost_€'] = df[' Cost_€'].round(2)
df['Revenue_€'] = df['Revenue_€'].round(2)


## 2.5 Finden und beheben Sie Typos.

#print(df['Month'].value_counts())
#print(df['Country'].value_counts())
#print(df['State'].value_counts())

df['Month'] = df['Month'].replace('Decmber', 'December')
df['Country'] = df['Country'].replace([' United States', 'United  States', 'United States ', ' United States '], 'United States')

#print(df['Month'].value_counts())
#print(df['Country'].value_counts())
#print(df['State'].value_counts())

#2.6. Finden Sie Ausreißer. Nutzen Sie hierfür das Box Plot und dokumentieren Sie das Diagramm zu jedem Feature.
# outlier, Boxplot, IQR
# Diagramm zu jedem Feature: Spalten (Features)
"""
numeric_cols = df.select_dtypes(include='number').columns

# Box Plot generate 
for column in numeric_cols:
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[column].dropna(), vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.grid(True)
    plt.show()

    # detect outloer
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    if not outliers.empty:
        print(f'Outliers detected in {column}:')
        print(outliers)
    else:
        print(f'No outliers detected in {column}.')
"""

#3. Speichern Sie den Datensatz unter dem Namen "bike_sales_clean.csv" zwischen.
df.to_csv('bike_sales_clean.csv', index=False)

#print(df.isnull().sum())
"""
#4. Data Visualization
    #4.1. Visualisieren Sie in einem geeignetem Diagram, wieviele Männer und wieviele Frauen ein Fahrrad gekauft haben.
plt.figure(figsize=(8, 6))
df.groupby('Customer_Gender')['Sales_Order #'].count().plot(kind='bar')
plt.title('Number of Bike Purchases by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=0)
plt.show()

    #4.2. Visualisieren Sie den Gewinn pro Land.
plt.figure(figsize=(10, 6))
df.groupby('Country')[' Profit_$ '].sum().plot(kind='bar')
plt.title('Total Profit by Country')
plt.xlabel('Country')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.show()

    #4.3. Visualisieren Sie wieviel Geld Frauen und Männer getrennt abhängig von ihrem Alter ausgeben (Customer_Age vs. Revenvue)
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Customer_Gender']=='M']['Customer_Age'], df[df['Customer_Gender']=='M']['Revenue_$'], label='Male', alpha=0.5)
plt.scatter(df[df['Customer_Gender']=='F']['Customer_Age'], df[df['Customer_Gender']=='F']['Revenue_$'], label='Female', alpha=0.5)
plt.title('Revenue vs. Customer Age by Gender')
plt.xlabel('Customer Age')
plt.ylabel('Revenue')
plt.legend()
plt.show()

"""
#print(df.shape)
#5. Data Codification, Wandeln Sie alle nicht-numerischen Features in numerische Features um.
# month, customer_gender, counttry, state, product_datagory, Sub_catagory -> nicht nummerlisch.     
# Label-Encoding or One-Hot-Encoding 

#one-hot encoding

df = pd.get_dummies(df)

#6. Speichern Sie den Datensatz unter dem Namen "bike_sales_codified.csv" zwischen.
df.to_csv('bike_sales_codified.csv', index=False)

#7. Data Reduction

    #df.drop_duplicates() beides Spalten und Zeilen
    #7.1. Löschen Sie redundante Features (spalten)

redundant_columns = df.columns[df.columns.duplicated()]
df.drop(redundant_columns, axis=1, inplace=True)

print(df.shape)

    #7.2. Suchen und löschen Sie redundante Records (Zeilen).
df.drop_duplicates(inplace=True)
print(df.shape)

    #7.3. Erstellen Sie eine Korrelationsmatrix (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) und dokumentieren diese.
# Erstellen einer Korrelationsmatrix
correlation_matrix = df.corr()
#print(correlation_matrix)

    #7.4. Löschen Sie ggf. Features die stark korrelieren und dokumentieren Sie Ihre Entscheidung.
threshold = 0.8  
highly_correlated_features = set()  
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
        
            highly_correlated_features.add(feature1)
            highly_correlated_features.add(feature2)

df.drop(highly_correlated_features, axis=1, inplace=True)
#print(correlation_matrix)

   
    #7.5. Führen Sie eine Principal Component Analysis durch (Achten Sie darauf, dass alle Features standardisiert werden müssen). Dokumentieren Sie die Anzahl der Principal Components, die 95% Varianz der Daten abdecken.

# Create and fit PCA model.
pca = PCA()  # Create PCA model.
pca.fit(scaled_data)  # Fit the PCA model to the scaled data.

# Check cumulative explained variance.
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)  # Check cumulative explained variance.

# Find the number of principal components explaining 95% of variance.
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # Find the number of principal components explaining 95% of variance.

# Refit PCA and transform principal components.
pca = PCA(n_components=n_components)  # Refit PCA with the updated number of components.
principal_components = pca.fit_transform(scaled_data)  # Transform the principal components.


#8. Speichern Sie den Datensatz, nicht standardisiert, unter dem Namen "bike_sales_reduced.csv" zwischen.
df.to_csv('bikw_sales_reduced.csv', index=False)

#9. Data Normailization, Normalisieren Sie alle Features
# Min-Max scaler init
scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(df)

# normalized data trancefer to DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

print(normalized_df.head())

#10. Speichern Sie den Datensatz unter dem Namen "bike_sales_normalized.csv".
df.to_csv('bike_sales_normalized.csv', index=False)
