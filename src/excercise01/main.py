import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Einlesen der Tabelle
car_price_ds = pd.read_csv("uncleaned2_bike_sales.csv")
'''
# create dataframe
df = pd.DataFrame(data=[[1, "Honda Civic", 290, 15000],
                        [2, "BMW", 300, 30000]], columns=["id", "CarName", "horsepower", "price"])

# # print entire dataframe
print("-- Dataframe --")
print(df)

# get first record of a dataframe
print("-- Get row by index --")
print(df.loc[0])

print("-- Set index --")
df = df.set_index("CarName")
print(df)

print("-- Get Row by row index --")
print(df.loc["BMW"])

print("-- Get Row by row number --")
print(df.iloc[:])

# get one column of a dataframe
print("-- Specific Column --")
print(df["horsepower"])

print("-- Description --")
print(df.describe())

print("-- Datatypes --")
print(df.dtypes)

# read a csv
car_price_ds = pd.read_csv("CarPrice_Assignment.csv")

# print car price dataset
print("-- Car Price Dataset --")
print(car_price_ds)

# print all column names
print("-- Car Price Dataset Column Names --")
print(car_price_ds.columns)


# get data from multiple columns
print("-- get data from multiple columns --")

price_vs_highwaympg = car_price_ds[["price", "highwaympg"]]
print("-- price vs highwaympg --")
print(price_vs_highwaympg)

print("-- Datatypes --")
print(price_vs_highwaympg.dtypes)

price_vs_highwaympg_np = price_vs_highwaympg.to_numpy()
print(type(price_vs_highwaympg_np))
print(price_vs_highwaympg_np.dtype)

print(price_vs_highwaympg_np.shape)

plt.scatter(price_vs_highwaympg_np[:, 0], price_vs_highwaympg_np[:, 1])
plt.show()
'''