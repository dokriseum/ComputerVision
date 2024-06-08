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
#print(missing_values)

print(df.tail())
print(df.columns)

def fill_missing_values(df, column, method):
    if method == 'Random Replacement':
        # Perform random replacement
        df[column].fillna(df[column].sample(n=df[column].isnull().sum(), replace=True), inplace=True)
    elif method == 'Fehlende Werte durch Querverweise ergänzen':
        pass
    elif method == 'Fill with zeros':
        # Fill missing values with zeros
        df[column].fillna(0, inplace=True)
    elif method == 'Fill with mean':
        # Fill missing values with mean
        df[column].fillna(df[column].mean(), inplace=True)
    elif method == 'stratified replacement':
        pass

    # If the column is in dollars, also fill the corresponding euro column
    if column.endswith('$'):
        euro_column = column.replace('$', '€')
        df[euro_column] = (df[column] * usd_to_euro).round(2)
        fill_missing_values(df, euro_column, method)

fill_missing_values(df, 'Day', 'Fill with mean')
fill_missing_values(df, 'Customer_Age', 'Fill with mean')
fill_missing_values(df, 'Customer_Gender', 'Random Replacement')
fill_missing_values(df, 'State', 'Fehlende Werte durch Querverweise ergänzen')
fill_missing_values(df, 'Order_Quantity', 'Fill with mean')
fill_missing_values(df, ' Unit_Cost_$', 'Fill with mean')
fill_missing_values(df, ' Unit_Price_$ ', 'Fill with mean')
fill_missing_values(df, ' Cost_$', 'Fill with mean')
fill_missing_values(df, 'Revenue_$', 'Fill with mean')
fill_missing_values(df, ' Profit_$ ', 'Fill with mean')

print(df.tail())


#€

## 2.5 Finden und beheben Sie Typos.

#print(df['Month'].value_counts())
#print(df['Country'].value_counts())
#print(df['State'].value_counts())

df['Month'] = df['Month'].replace('Decmber', 'December')
df['Country'] = df['Country'].replace([' United States', 'United  States'], 'United States')

#print(df['Month'].value_counts())
#print(df['Country'].value_counts())


#2.6. Finden Sie Ausreißer. Nutzen Sie hierfür das Box Plot und dokumentieren Sie das Diagramm zu jedem Feature.

#3. Speichern Sie den Datensatz unter dem Namen "bike_sales_clean.csv" zwischen.

#4. Data Visualization

    #4.1. Visualisieren Sie in einem geeignetem Diagram, wieviele Männer und wieviele Frauen ein Fahrrad gekauft haben.

    #4.2. Visualisieren Sie den Gewinn pro Land.

    #4.3. Visualisieren Sie wieviel Geld Frauen und Männer getrennt abhängig von ihrem Alter ausgeben (Customer_Age vs. Revenvue)

#5. Data Codification, Wandeln Sie alle nicht-numerischen Features in numerische Features um.

#6. Speichern Sie den Datensatz unter dem Namen "bike_sales_codified.csv" zwischen.

#7. Data Reduction

    #7.1. Löschen Sie redundante Features

    #7.2. Suchen und löschen Sie redundante Records (Zeilen).

    #7.3. Erstellen Sie eine Korrelationsmatrix (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) und dokumentieren diese.

    #7.4. Löschen Sie ggf. Features die stark korrelieren und dokumentieren Sie Ihre Entscheidung.

    #7.5. Führen Sie eine Principal Component Analysis durch (Achten Sie darauf, dass alle Features standardisiert werden müssen). Dokumentieren Sie die Anzahl der Principal Components, die 95% Varianz der Daten abdecken.

#8. Speichern Sie den Datensatz, nicht standardisiert, unter dem Namen "bike_sales_reduced.csv" zwischen.

#9. Data Normailization, Normalisieren Sie alle Features

#10. Speichern Sie den Datensatz unter dem Namen "bike_sales_normalized.csv".

