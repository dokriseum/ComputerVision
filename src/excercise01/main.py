import pandas as pd


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

# Define the exchange rate from USD to Euro
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
threshold_col = len(df) * 0.4  
df = df.dropna(thresh=threshold_col, axis=1)


## 2.3   Löschen Sie Zeilen (Records) in denen mehr als 60% der Einträge fehlen.

# Calculate threshold for rows and drop rows with more than 60% missing values
threshold_row = len(df.columns) * 0.4  
df = df.dropna(thresh=threshold_row, axis=0)

#print(df.shape)

## 2.4 Ergänzen Sie die restlichen fehlenden Einträge mit einer der folgenden Methoden
missing_values = df.isnull().sum()
print(missing_values)

print(df.tail())

def fill_missing_values(df, column, method):
    if method == 'Random Replacement':
        # Perform random replacement
        df[column].fillna(df[column].sample(n=df[column].isnull().sum(), replace=True), inplace=True)
    elif method == 'Fehlende Werte durch Querverweise ergänzen':
        # Perform cross-reference replacement
        # You would need to replace missing values based on some other column(s) in the DataFrame
        # For example:
        # df[column].fillna(df['Some_Other_Column'], inplace=True)
        pass
    elif method == 'Fill with zeros':
        # Fill missing values with zeros
        df[column].fillna(0, inplace=True)
    elif method == 'Fill with mean':
        # Fill missing values with mean
        df[column].fillna(df[column].mean(), inplace=True)
    elif method == 'stratified replacement':
        # Perform stratified replacement
        # You would need to implement the logic for this method
        pass

fill_missing_values(df, 'Customer_Gender', 'Random Replacement')
fill_missing_values(df, 'State', 'Fehlende Werte durch Querverweise ergänzen')
fill_missing_values(df, 'Order_Quantity', 'Fill with mean')
fill_missing_values(df, ' Unit_Cost_$', 'Fill with mean')
fill_missing_values(df, ' Unit_Price_$ ', 'Fill with mean')
fill_missing_values(df, ' Unit_Cost_$', 'Fill with mean')
fill_missing_values(df, ' Unit_Price_$ ', 'Fill with mean')
fill_missing_values(df, ' Profit_$ ', 'Fill with mean')
fill_missing_values(df, ' Cost_$', 'Fill with mean')
fill_missing_values(df, 'Revenue_$', 'Fill with mean')
# Add more calls for other columns as needed

df[' Unit_Cost_$'] = df[' Unit_Cost_$'].round(2)
df[' Unit_Price_$ '] = df[' Unit_Price_$ '].round(2)
df[' Profit_$ '] = df[' Profit_$ ' ].round(2)
df[' Cost_$'] = df[' Cost_$'].round(2)

print(df.tail())


## 2.5 Finden und beheben Sie Typos.

print(df['Month'].value_counts())
print(df['Country'].value_counts())
print(df['State'].value_counts())

df['Month'] = df['Month'].replace('Decmber', 'December')
df['Country'] = df['Country'].replace([' United States', 'United  States'], 'United States')

print(df['Month'].value_counts())
print(df['Country'].value_counts())