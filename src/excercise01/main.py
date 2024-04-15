import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os

os_path = os.path.dirname(os.path.abspath(__file__))
# Einlesen der Tabelle
data_raw = pd.read_csv(os.path.join(os_path, "uncleaned2_bike_sales.csv"))

print(data_raw)


# Berechnen der Schwellenwerte
thresh_row = int(0.4 * len(data_raw.columns))  # Mindestanzahl von nicht-fehlenden Werten pro Zeile
thresh_col = int(0.4 * len(data_raw))          # Mindestanzahl von nicht-fehlenden Werten pro Spalte

# Entfernen von Spalten mit mehr als 60% fehlenden Werten
df_cleaned_cols = data_raw.dropna(axis=1, thresh=thresh_col)

# Ausgabe der bereinigten Tabelle (nur Spalte)
print("-- df_cleaned_cols --")
print(df_cleaned_cols)

# Entfernen von Zeilen mit mehr als 60% fehlenden Werten
df_cleaned_rows = df_cleaned_cols.dropna(axis=0, thresh=thresh_row)

# Ausgabe der bereinigten Tabelle (mit Zeilen)
print("-- df_cleaned_rows --")
print(df_cleaned_rows)

data_preclean = df_cleaned_rows.to_csv(os.path.join(os_path, "bike_sales_preclean.csv"), index=False)

'''
# Ersetzen der fehlenden Werte in einer Spalte durch zufällige Werte aus derselben Spalte
df_cleaned_rows = df_cleaned_rows[column_name] = df_cleaned_rows[column_name].apply(
    lambda x: np.random.choice(df_cleaned_rows[column_name].dropna().values) if pd.isnull(x) else x
)
'''

column_name = 'Customer_Age'
# Extrahiere die vorhandenen Altersdaten, die nicht fehlen
available_ages = df_cleaned_rows[column_name].dropna().unique()
# Ersetze die fehlenden Werte in der Spalte 'Kunde_Alter' durch zufällige Werte aus derselben Spalte
df_cleaned_rows.loc[:, column_name] = df_cleaned_rows[column_name].apply(
    lambda x: np.random.choice(available_ages) if pd.isnull(x) else x
)

# Geschlecht ausfüllen
column_name = 'Customer_Gender'
available_sex = df_cleaned_rows[column_name].dropna().unique()  # Extrahiere die vorhandenen Werte (M und W)
# Sicherstellen, dass die Operation auf dem Original DataFrame ausgeführt wird
df_cleaned_rows.loc[:, column_name] = df_cleaned_rows.loc[:, column_name].apply(
    lambda x: np.random.choice(available_sex) if pd.isnull(x) else x
)

# Keine Notwendigkeit für eine zusätzliche Zuweisung, wenn nicht benötigt
data_clean = df_cleaned_rows


data_clean = data_clean.to_csv(os.path.join(os_path, "bike_sales_clean.csv"), index=False)

print(data_clean)

print("-- Ende --")



'''


# plot function
x = np.linspace(0, 10, num=1000)
y = np.cos(x)

plt.plot(x, y)
plt.show()

# plot matrix
random_matrix = np.random.random(size=(2, 1000))

print(random_matrix.shape)
print(random_matrix[0].shape)

plt.plot(random_matrix)
plt.show()

plt.scatter(random_matrix[0], random_matrix[1])
plt.show()

plt.hist(random_matrix[0])
plt.show()
# exit()

# Subplots
fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
fig.tight_layout()
axs[0].plot(random_matrix[0], random_matrix[1])
axs[0].set_xlabel("x axis")
axs[0].set_ylabel("y axis")
axs[0].set_title("plot random values")
axs[1].scatter(random_matrix[0], random_matrix[1])
axs[1].set_title("scatter random values")
axs[2].hist(random_matrix[0])

plt.show()

######################################
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