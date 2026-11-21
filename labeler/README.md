# Socken-Bilder labeln und YOLO trainieren

Lokales Web-Tool in Go zum Labeln von Bildern und Starten/Überwachen eines Ultralytics-YOLO-Trainings. Es zeichnet YOLO-Bounding-Boxes (Klasse `0 = sock`), speichert die Labels als Textdateien und bietet Eingabefelder für Modell, Dataset, Epochen, Bildgröße, Batch, Device, Patience und Worker. Laufstatus, Trainingslogs und Graphen für Precision/Recall/mAP/Loss werden aus `results.csv` angezeigt.

## Voraussetzung

- Go 1.22+
- Python mit Ultralytics installiert und das `yolo`-Kommando im `PATH`:

```sh
python3 -m pip install ultralytics
```

Falls `yolo` nicht im PATH liegt, setze `YOLO_BIN` auf den Pfad zur Ultralytics-CLI. Zum Beispiel bei einer virtuellen Umgebung: `YOLO_BIN=/path/to/.venv/bin/yolo`.

## Starten

```sh
mkdir -p images labels
# JPG-, PNG- oder WebP-Bilder nach images/ kopieren
cd labeler
go run .
```

Öffne danach `http://127.0.0.1:8080`. Die Anwendung bindet standardmäßig ausschließlich an localhost. Trainingsdaten und Ergebnisse bleiben auf diesem Rechner. Bilder und Labels werden relativ zum Repository-Ordner in `images/` und `labels/` erwartet, wenn die App aus `labeler/` gestartet wird.

## Dataset für YOLO erstellen

Der Labeler legt YOLO-Dateien in `labels/` ab. Für Training/Validierung müssen Bilder und gleichnamige Labeldateien in YOLO-Struktur liegen:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

Beispiel `dataset.yaml` (absolute Pfade vermeiden Pfadverwechslungen):

```yaml
path: /absoluter/pfad/zum/dataset
train: images/train
val: images/val
names:
  0: sock
```

Teile die Bilder zufällig in Train und Val auf, z. B. 80/20. Die Validierungsbilder sollten nicht dieselben Aufnahmen wie Train enthalten. Labels haben die Form `class_id x_center y_center width height`, normiert von 0 bis 1.

## Training bedienen

Im Abschnitt „YOLOv8n-Training“ stellst du Dataset-YAML, Checkpoint (Standard `yolov8n.pt`), Epochen, `imgsz`, Batch, Device (`0`, `cpu`, `mps`), Patience und Worker ein. Klick auf „Training starten“. Das Go-Programm startet das `yolo detect train`-Kommando mit diesen validierten Parametern. Während des Trainings kannst du den Status/Log beobachten oder es stoppen.

Ergebnisse liegen in `runs/detect/sock-<Zeitstempel>/`, unter anderem:

- `weights/best.pt` und `weights/last.pt`
- `results.csv` — Trainings- und Validierungsmetriken
- Ultralytics-Plot- und Konfigurationsdateien

Der UI-Graph liest die CSV-Metriken aus dem aktuellen Lauf. Wähle im Menü mAP50, mAP50-95, Precision, Recall oder Loss. Die Weboberfläche aktualisiert Status/Graph während des Trainings.

## Sicherheitshinweis

Das Training kann hohe CPU-/GPU-Auslastung verursachen. Die Webapp ist für lokale Nutzung gedacht; nicht öffentlich ins Netzwerk exponieren. Start-/Stop-Endpunkte haben keine Authentifizierung und sind daher absichtlich nur auf `127.0.0.1` gebunden. Trainingsgewichte und Dataset bleiben lokal.
