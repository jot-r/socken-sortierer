package main

import (
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

//go:embed index.html
var webFiles embed.FS

type Label struct {
	Class int     `json:"class"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	W     float64 `json:"w"`
	H     float64 `json:"h"`
}

type App struct {
	imagesDir string
	labelsDir string
}

func main() {
	imagesDir := filepath.Clean("../images")
	labelsDir := filepath.Clean("../labels")
	addr := "127.0.0.1:8080"
	if value := os.Getenv("LABELER_IMAGES"); value != "" { imagesDir = filepath.Clean(value) }
	if value := os.Getenv("LABELER_LABELS"); value != "" { labelsDir = filepath.Clean(value) }
	if value := os.Getenv("LABELER_ADDR"); value != "" { addr = value }

	if err := os.MkdirAll(imagesDir, 0755); err != nil { log.Fatal(err) }
	if err := os.MkdirAll(labelsDir, 0755); err != nil { log.Fatal(err) }

	app := &App{imagesDir: imagesDir, labelsDir: labelsDir}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/images", app.handleImages)
	mux.HandleFunc("/api/labels", app.handleLabels)
	mux.HandleFunc("/images/", app.handleImage)
	mux.HandleFunc("/", app.handleIndex)

	server := &http.Server{
		Addr:              addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("Sock labeling tool: http://%s", addr)
	log.Printf("Images: %s | Labels: %s", app.imagesDir, app.labelsDir)
	log.Fatal(server.ListenAndServe())
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func (a *App) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" { http.NotFound(w, r); return }
	if r.Method != http.MethodGet { methodNotAllowed(w); return }
	data, err := webFiles.ReadFile("index.html")
	if err != nil { http.Error(w, "Could not load UI", http.StatusInternalServerError); return }
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (a *App) handleImages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet { methodNotAllowed(w); return }
	entries, err := os.ReadDir(a.imagesDir)
	if err != nil { http.Error(w, "Could not read images directory", http.StatusInternalServerError); return }
	images := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !isImage(entry.Name()) { continue }
		images = append(images, entry.Name())
	}
	sort.Strings(images)
	writeJSON(w, images)
}

func (a *App) handleImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet { methodNotAllowed(w); return }
	name, ok := safeImageName(strings.TrimPrefix(r.URL.Path, "/images/"))
	if !ok { http.Error(w, "Invalid image name", http.StatusBadRequest); return }
	path := filepath.Join(a.imagesDir, name)
	if _, err := os.Stat(path); err != nil { http.NotFound(w, r); return }
	w.Header().Set("Cache-Control", "no-store")
	http.ServeFile(w, r, path)
}

func (a *App) handleLabels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		name, ok := safeImageName(r.URL.Query().Get("image"))
		if !ok { http.Error(w, "Invalid image name", http.StatusBadRequest); return }
		labels, err := a.readLabels(name)
		if err != nil { http.Error(w, "Could not read labels: "+err.Error(), http.StatusInternalServerError); return }
		writeJSON(w, labels)
	case http.MethodPut:
		defer r.Body.Close()
		var request struct {
			Image  string  `json:"image"`
			Labels []Label `json:"labels"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil { http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest); return }
		if err := ensureJSONEnd(decoder); err != nil { http.Error(w, "Invalid JSON body", http.StatusBadRequest); return }
		name, ok := safeImageName(request.Image)
		if !ok { http.Error(w, "Invalid image name", http.StatusBadRequest); return }
		if _, err := os.Stat(filepath.Join(a.imagesDir, name)); err != nil { http.Error(w, "Image not found", http.StatusNotFound); return }
		if len(request.Labels) > 5000 { http.Error(w, "Too many labels", http.StatusBadRequest); return }
		for _, label := range request.Labels {
			if label.Class != 0 || !finite01(label.X) || !finite01(label.Y) || !finite01(label.W) || !finite01(label.H) || label.W <= 0 || label.H <= 0 || label.X-label.W/2 < -1e-6 || label.Y-label.H/2 < -1e-6 || label.X+label.W/2 > 1.000001 || label.Y+label.H/2 > 1.000001 {
				http.Error(w, "Labels must be class 0 with normalized center/size coordinates between 0 and 1", http.StatusBadRequest)
				return
			}
		}
		if err := a.writeLabels(name, request.Labels); err != nil { http.Error(w, "Could not save labels", http.StatusInternalServerError); return }
		writeJSON(w, map[string]any{"saved": true, "count": len(request.Labels)})
	default:
		methodNotAllowed(w)
	}
}

func (a *App) readLabels(imageName string) ([]Label, error) {
	labelName := strings.TrimSuffix(imageName, filepath.Ext(imageName)) + ".txt"
	path := filepath.Join(a.labelsDir, labelName)
	contents, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) { return []Label{}, nil }
	if err != nil { return nil, err }
	labels := make([]Label, 0)
	for lineNumber, line := range strings.Split(string(contents), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 { continue }
		if len(fields) != 5 { return nil, fmt.Errorf("line %d should contain 5 values", lineNumber+1) }
		var label Label
		if _, err := fmt.Sscanf(strings.Join(fields, " "), "%d %f %f %f %f", &label.Class, &label.X, &label.Y, &label.W, &label.H); err != nil { return nil, fmt.Errorf("line %d: %w", lineNumber+1, err) }
		labels = append(labels, label)
	}
	return labels, nil
}

func (a *App) writeLabels(imageName string, labels []Label) error {
	labelName := strings.TrimSuffix(imageName, filepath.Ext(imageName)) + ".txt"
	var content strings.Builder
	for _, label := range labels {
		fmt.Fprintf(&content, "%d %.6f %.6f %.6f %.6f\n", label.Class, label.X, label.Y, label.W, label.H)
	}
	return os.WriteFile(filepath.Join(a.labelsDir, labelName), []byte(content.String()), 0644)
}

func safeImageName(name string) (string, bool) {
	if name == "" || name != filepath.Base(name) || strings.ContainsAny(name, `/\\`) || !isImage(name) { return "", false }
	return name, true
}

func isImage(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png", ".webp", ".bmp": return true
	default: return false
	}
}

func finite01(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1 }

func ensureJSONEnd(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) { return errors.New("extra data") }
	return nil
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(value)
}

func methodNotAllowed(w http.ResponseWriter) {
	w.Header().Set("Allow", "GET, PUT")
	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}
