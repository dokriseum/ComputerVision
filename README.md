# CV_Gruppe1

Folgende Anleitung ist nur für unix-basierten Systeme (GNU/Linux oder macOS).

## Einleitung

In diesem Projekt wurden 3 Klassen getestet, die i.d.R. in der Öffentlichkeit zu sehen sind. Diese sind: Fahrrad (bike), Mülleimer (bin) und Straßennamensschild (shield).

### Begriffserklärung

xc = xception

## Einrichtung

Zur Erstellung des virtuellen Entwicklungsumgebung in Python:

```bash
python3.11 -m venv venv
```

aktivieren

```bash
source .venv/bin/activate
```

und installieren der requirements

```bash
pip install -r requirements.txt
```

## Ausführung

### Trainings- und Validierungsskript

1. zu erst in den jeweiligen Ordner navigieren, bspw. `cd XC_FALL2`
2. falls nicht schon geschehen in die Entwicklungsumgebung wechseln mit `source .venv/bin/activate`
3. anschließend das Skript starten `python3 ./xc_fall2.py $(pwd)`

## Live Vorführung

1. falls nicht schon geschehen in die Entwicklungsumgebung wechseln mit `source .venv/bin/activate`
2. anschließend das Skript starten `python3 Read_Camera_Image_And_Prognose_Object.py`