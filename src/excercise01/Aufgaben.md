

## Führen Sie Data Cleaning auf dem Datensatz "uncleaned2_bike_sales.csv" durch.

----
1. Verschaffen Sie sich einen Überblick über die Daten.
----
2. Data Cleaning
    1. Löschen Sie Spalten (Features) in denen mehr als 60% der Einträge fehlen.
    2. Löschen Sie Zeilen (Records) in denen mehr als 60% der Einträge fehlen.
    3. Ergänzen Sie die restlichen fehlenden Einträge mit einer der folgenden Methoden. Dokumentieren und begründen Sie, welche Methode Sie für welches Feature verwendet haben.
        - Random Replacement
        - Fehlende Werte durch Querverweise ergänzen
        - Fill with zeros
        - Fill with mean
        - stratified replacement
    4. Finden und beheben Sie Typos.
    5. Finden Sie Ausreißer. Nutzen Sie hierfür das Box Plot und dokumentieren Sie das Diagramm zu jedem Feature.
    6. Konvertieren Sie die Preise in € nach $. Recherchieren Sie hierfür den zu dieser Zeit aktuellen Kurs und dokumentieren Sie diesen.
----
3. Speichern Sie den Datensatz unter dem Namen "bike_sales_clean.csv" zwischen.
----
4. Data Visualization
    1. Visualisieren Sie in einem geeignetem Diagram, wieviele Männer und wieviele Frauen ein Fahrrad gekauft haben.
    2. Visualisieren Sie den Gewinn pro Land.
    3. Visualisieren Sie wieviel Geld Frauen und Männer getrennt abhängig von ihrem Alter ausgeben (Customer_Age vs. Revenvue)
----
5. Data Codification
    1. Wandeln Sie alle nicht-numerischen Features in numerische Features um.
----
6. Speichern Sie den Datensatz unter dem Namen "bike_sales_codified.csv" zwischen.
----
7. Data Reduction
    1. Löschen Sie redundante Features
    2. Suchen und löschen Sie redundante Records (Zeilen).
    3. Erstellen Sie eine Korrelationsmatrix (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) und dokumentieren diese.
    4. Löschen Sie ggf. Features, die stark korrelieren und dokumentieren Sie Ihre Entscheidung.
    5. Führen Sie eine Principal Component Analysis durch (Achten Sie darauf, dass alle Features standardisiert werden müssen). Dokumentieren Sie die Anzahl der Principal Components, die 95% Varianz der Daten abdecken.
----
8. Speichern Sie den Datensatz, nicht standardisiert, unter dem Namen "bike_sales_reduced.csv" zwischen.
----
9. Data Normailization
    1. Normalisieren Sie alle Features
----
10. Speichern Sie den Datensatz unter dem Namen "bike_sales_normalized.csv".


