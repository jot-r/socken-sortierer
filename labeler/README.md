# Socken-Bilder für YOLO labeln

Ein kleines lokales Web-Labeling-Tool in Go. Es zeigt Bilder aus `images/`, erlaubt per Maus oder Touch YOLO-Bounding-Boxes für eine Klasse (`0 = sock`) zu zeichnen und speichert sie als YOLO-Textdateien nach `labels/`.

## Starten

Voraussetzung: Go 1.22 oder neuer.

```sh
mkdir -p images labels
# JPG-, PNG- oder WebP-Bilder in images/ kopieren
cd labeler
go run .
```

Dann auf demselben Rechner `http://127.0.0.1:8080` öffnen. Das Tool bindet standardmäßig nur an localhost und braucht weder Cloud noch externe JavaScript-Bibliotheken. `images/` und `labels/` sind relativ zum Arbeitsverzeichnis des Programms (bei obigen Befehlen also Repository-Root).

Optional lassen sich Verzeichnisse und Adresse konfigurieren:

```sh
LABELER_IMAGES=/pfad/zu/bildern LABELER_LABELS=/pfad/zu/labels go run .
```

## Bedienung

- Ziehe einen Rahmen um jede Socke.
- Tippe/klicke einen bestehenden Rahmen an, um ihn zu löschen.
- „Letztes löschen“ entfernt das zuletzt angelegte Label.
- Speichere mit dem Button oder `Strg/Cmd+S`.
- Mit Pfeil links/rechts wechselst du zwischen Bildern.

Die Textdatei hat denselben Basisnamen wie das Bild und das YOLO-Detection-Format:

```text
class_id x_center y_center width height
```

Koordinaten sind auf 0–1 normiert; die einzige Klasse hat ID `0` und heißt `sock`. Beispiel für Ultralytics-Dataset-YAML:

```yaml
path: /absoluter/pfad/zum/dataset
train: images/train
val: images/val
names:
  0: sock
```

Lege dafür Bilder und Labeldateien entsprechend in `images/train`, `images/val` und `labels/train`, `labels/val`. Das Tool selbst liest alle unterstützten Bilder aus einem einzigen Eingabeordner und speichert Labels zentral in einem Labelordner.

## Hinweise

Die Labels werden nur lokal gespeichert. Das Tool speichert erst beim Klick auf „Speichern“; beim Verlassen mit ungespeicherten Änderungen erscheint eine Warnung. Unterstützte Formate: JPG/JPEG, PNG, WebP und BMP. Bitte nicht an ein öffentliches Netzwerk binden: Das lokale Tool hat keine Authentifizierung.
