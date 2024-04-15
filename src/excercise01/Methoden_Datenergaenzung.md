# Methoden zur Ergänzung fehlender Daten
Beim Umgang mit fehlenden Daten in Datensätzen gibt es verschiedene Methoden, um diese zu ersetzen oder zu ergänzen. Jede Methode hat spezifische Vor- und Nachteile und eignet sich besser für bestimmte Arten von Daten oder Analysezielen. Hier erkläre ich die fünf genannten Methoden und gebe Beispiele für ihre Anwendung:

### Methode 1: Random Replacement

**Erklärung:** Bei dieser Methode werden fehlende Werte durch zufällig ausgewählte Werte aus derselben Spalte ersetzt. Dies kann sinnvoll sein, um die ursprüngliche Verteilung der Daten beizubehalten.

**Einsatzbeispiel:** Wenn Sie eine Spalte mit Kategorien haben (z. B. Farben eines Produkts), und Sie möchten die Verteilung der Kategorien nicht verzerren, können Sie fehlende Werte durch zufällige vorhandene Kategorien aus derselben Spalte ersetzen.

### Methode 2: Fehlende Werte durch Querverweise ergänzen

**Erklärung:** Diese Methode verwendet Informationen aus anderen Spalten (Querverweise), um die fehlenden Werte zu ersetzen. Dies ist nützlich, wenn es eine logische oder statistische Verbindung zwischen den Spalten gibt.

**Einsatzbeispiel:** Wenn in einem Datensatz die Spalte für das Geburtsdatum und das Alter einer Person fehlt, können Sie das fehlende Alter basierend auf dem Geburtsdatum und dem aktuellen Datum ergänzen.

### Methode 3: Fill with zeros

**Erklärung:** Hierbei werden alle fehlenden Werte einer Spalte durch Null ersetzt. Dies ist besonders in numerischen Datenfeldern anwendbar, wo Null einen sinnvollen Wert darstellen kann (z.B. keine Einnahmen, keine Teilnahme).

**Einsatzbeispiel:** In einem Datensatz über Verkäufe könnten fehlende Werte in der Spalte "Anzahl der verkauften Einheiten" mit Null ersetzt werden, was bedeutet, dass keine Verkäufe stattgefunden haben.

### Methode 4: Fill with mean

**Erklärung:** Fehlende Werte werden durch den Mittelwert (Durchschnitt) der vorhandenen Werte in derselben Spalte ersetzt. Dies ist hilfreich, um die zentralen Tendenzen der Daten nicht zu stören.

**Einsatzbeispiel:** In einem Datensatz mit Testergebnissen, wo einige Werte fehlen, kann der Mittelwert der restlichen Noten verwendet werden, um eine realistische Schätzung der fehlenden Werte zu bieten.

### Methode 5: Stratified Replacement

**Erklärung:** Bei dieser Methode werden fehlende Werte basierend auf einer Kategorisierung (Stratifikation) der Daten ersetzt. Dies kann z.B. der Durchschnittswert innerhalb einer bestimmten Kategorie sein.

**Einsatzbeispiel:** In einem Gehaltsdatensatz könnten fehlende Gehälter basierend auf der Abteilung und der Berufsbezeichnung ergänzt werden, indem der Durchschnitt der Gehälter innerhalb jeder spezifischen Gruppe (Stratum) verwendet wird.

# Fazit

Die Wahl der Methode hängt stark von der Art der Daten und dem Kontext der Analyse ab. Es ist wichtig, die Implikationen jeder Methode zu verstehen und wie sie die Ergebnisse Ihrer Datenanalyse beeinflussen kann. In der Praxis ist es oft sinnvoll, mehrere Methoden zu testen und ihre Auswirkungen auf die Analyseergebnisse zu vergleichen.

----

- Customer_Age => Random Replacement (falls Durchschnitt errechnen)
- Customer_Gender => Random Replacement (falls Durchschnitt errechnen)
- State => Random Replacement && fehlende Werte durch Querverweise ergänzen (Bundesstaat gehört immer zu einem Land, deshalb )
- Product_Category => fehlende Werte durch Querverweise ergänzen (gefüllte Produktkategorie sind gleich)
- Sub_Category => fehlende Werte durch Querverweise ergänzen (gefüllte Unterproduktkategorie sind gleich)
- Order_Quantity => 
- Unit_Cost_$ => 
- Unit_Price_$ => 
- Profit_$ => 
- Cost_$ => 
- Revenue_$ => 