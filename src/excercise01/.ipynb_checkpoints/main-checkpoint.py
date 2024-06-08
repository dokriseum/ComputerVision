# 1. Datenüberblick verschaffen. 

import pandas as pd
import random


#csv file path
file_path = '/home/soyoung/cvss-24-gruppe-1/src/excercise01/uncleaned2_bike_sales.csv'

#load data
df = pd.read_csv(file_path)

#first satz von data read
#print("Erste fünf Zeilen des Datensatz: ")
#print(df.head())

#Datensatz info
#print("Info zum Datensatz:")
#print(df.info())

#anzahl der spalte
#print("Anzahl der Spalte:")
print(df.isnull().sum())

#2. Data Cleaning
## 2.1 kovertieren in Euro nach dollar

# set exchange rate 
#exchange_rate = 1.1

# exchagne price
#df['Price'] = df['Price'].apply(lambda x: x * exchange_rate if pd.notnull(x) else x)

#random = random  # keep randomness

#print(df.iloc(0))

#print(df.shape)
#print(df.columns)
#print(df.index)
#print(df.values)
#print(df.describe())
#print(df['Unit'])

#3. Löschen Sie Spalten (Features) in denen mehr als 60% der Einträge fehlen.

#4  Löschen Sie Zeilen (Records) in denen mehr als 60% der Einträge fehlen.