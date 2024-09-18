#!/bin/bash

# Überprüfen, ob genügend Argumente übergeben wurden
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <source_directory> <destination_directory> <word>"
  exit 1
fi

# Variablen für Quell- und Zielordner sowie das zu verwendende Wort
SOURCE_DIR="$1"
DEST_DIR="$2"
WORD="$3"

# Überprüfen, ob der Quellordner existiert
if [ ! -d "$SOURCE_DIR" ];then
  echo "Der Quellordner existiert nicht: $SOURCE_DIR"
  exit 1
fi

# Erstellen des Zielordners, falls er nicht existiert
mkdir -p "$DEST_DIR"

# Zählen der Bilddateien im Quellordner und allen Unterordnern (einschließlich Rohbild-Dateien)
file_count=$(find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.tif" -o -iname "*.heic" -o -iname "*.webp" -o -iname "*.dng" -o -iname "*.nef" -o -iname "*.cr2" \) | wc -l)

# Initialisieren der Fortschrittsanzeige und Zähler für die Dateinummer
current_file=0
file_number=1

# Schleife durch alle Bilddateien im Quellordner und allen Unterordnern (einschließlich Rohbild-Dateien)
find "$SOURCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.tif" -o -iname "*.heic" -o -iname "*.webp" -o -iname "*.dng" -o -iname "*.nef" -o -iname "*.cr2" \) | while read -r file; do
  # Dateiname und Zielpfad setzen
  new_filename="${WORD}_$(printf "%03d" $file_number).png"
  destination="$DEST_DIR/$new_filename"

  # Konvertierung in PNG mit ImageMagick und Reduktion der Dateigröße auf maximal 5MB
  magick "$file" -resize 3000x3000\> -quality 85 -define png:compression-level=9 -define png:exclude-chunk=all "$destination"

  # Überprüfen der Dateigröße und iteratives Reduzieren falls nötig
  while [ $(stat -f%z "$destination") -gt 5242880 ]; do
    magick "$destination" -resize 90% -quality 85 "$destination"
  done

  # Überprüfen, ob die Konvertierung erfolgreich war
  if [ $? -eq 0 ];then
    # Aktualisierung der Fortschrittsanzeige und Erhöhung der Dateinummer
    current_file=$((current_file + 1))
    file_number=$((file_number + 1))
    echo "[$current_file/$file_count] Konvertiert: $file -> $destination"
  else
    echo "Fehler bei der Konvertierung von: $file"
  fi
done

echo "Alle Dateien wurden konvertiert, verkleinert und im Zielordner gespeichert: $DEST_DIR"
