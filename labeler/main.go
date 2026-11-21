package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
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

type TrainConfig struct {
	Data    string `json:"data"`
	Model   string `json:"model"`
	Epochs  int    `json:"epochs"`
	ImgSize int    `json:"imgsz"`
	Batch   int    `json:"batch"`
	Device  string `json:"device"`
	Patience int   `json:"patience"`
	Workers int    `json:"workers"`
}

type MetricPoint map[string]float64

type App struct {
	imagesDir string
	labelsDir string
	rootDir   string
	trainMu   sync.Mutex
	training  trainingState
}

type trainingState struct {
	Running  bool      `json:"running"`
	Status   string    `json:"status"`
	Error    string    `json:"error,omitempty"`
	Logs     []string  `json:"logs"`
	RunDir   string    `json:"-"`
	Started  time.Time `json:"started,omitempty"`
	Finished time.Time `json:"finished,omitempty"`
	cancel   context.CancelFunc
}

var modelPattern = regexp.MustCompile(`^[A-Za-z0-9_./:\\-]+\.pt$`)
var devicePattern = regexp.MustCompile(`^(cpu|mps|[0-9]+(,[0-9]+)*)$`)

func main() {
	cwd, err := os.Getwd()
	if err != nil { log.Fatal(err) }
	rootDir := cwd
	if filepath.Base(cwd) == "labeler" { rootDir = filepath.Dir(cwd) }
	imagesDir := filepath.Join(rootDir, "images")
	labelsDir := filepath.Join(rootDir, "labels")
	addr := "127.0.0.1:8080"
	if value := os.Getenv("LABELER_IMAGES"); value != "" { imagesDir = filepath.Clean(value) }
	if value := os.Getenv("LABELER_LABELS"); value != "" { labelsDir = filepath.Clean(value) }
	if value := os.Getenv("LABELER_ADDR"); value != "" { addr = value }
	if err := os.MkdirAll(imagesDir, 0755); err != nil { log.Fatal(err) }
	if err := os.MkdirAll(labelsDir, 0755); err != nil { log.Fatal(err) }

	app := &App{imagesDir: imagesDir, labelsDir: labelsDir, rootDir: rootDir}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/images", app.handleImages)
	mux.HandleFunc("/api/labels", app.handleLabels)
	mux.HandleFunc("/api/training", app.handleTraining)
	mux.HandleFunc("/api/training/stop", app.handleStopTraining)
	mux.HandleFunc("/api/metrics", app.handleMetrics)
	mux.HandleFunc("/images/", app.handleImage)
	mux.HandleFunc("/", app.handleIndex)
	server := &http.Server{Addr: addr, Handler: logRequests(mux), ReadHeaderTimeout: 5 * time.Second}
	log.Printf("Sock labeling and training tool: http://%s", addr)
	log.Printf("Root: %s | Images: %s | Labels: %s", rootDir, app.imagesDir, app.labelsDir)
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
	for _, entry := range entries { if !entry.IsDir() && isImage(entry.Name()) { images = append(images, entry.Name()) } }
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
		var request struct { Image string `json:"image"`; Labels []Label `json:"labels"` }
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
				http.Error(w, "Labels must be class 0 with normalized center/size coordinates between 0 and 1", http.StatusBadRequest); return
			}
		}
		if err := a.writeLabels(name, request.Labels); err != nil { http.Error(w, "Could not save labels", http.StatusInternalServerError); return }
		writeJSON(w, map[string]any{"saved": true, "count": len(request.Labels)})
	default: methodNotAllowed(w)
	}
}

func (a *App) handleTraining(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		a.trainMu.Lock()
		state := a.training
		state.cancel = nil
		if state.Logs == nil { state.Logs = []string{} }
		a.trainMu.Unlock()
		writeJSON(w, state)
	case http.MethodPost:
		defer r.Body.Close()
		var config TrainConfig
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&config); err != nil { http.Error(w, "Invalid training settings: "+err.Error(), http.StatusBadRequest); return }
		if err := ensureJSONEnd(decoder); err != nil { http.Error(w, "Invalid JSON body", http.StatusBadRequest); return }
		if err := a.validateTrainConfig(&config); err != nil { http.Error(w, err.Error(), http.StatusBadRequest); return }
		yoloBin := os.Getenv("YOLO_BIN")
		if yoloBin == "" { yoloBin = "yolo" }
		if _, err := exec.LookPath(yoloBin); err != nil { http.Error(w, "Ultralytics CLI 'yolo' not found in PATH. Install it with: python -m pip install ultralytics", http.StatusPreconditionFailed); return }
		a.trainMu.Lock()
		if a.training.Running { a.trainMu.Unlock(); http.Error(w, "A training run is already active", http.StatusConflict); return }
		ctx, cancel := context.WithCancel(context.Background())
		runName := "sock-" + time.Now().Format("20060102-150405")
		runDir := filepath.Join(a.rootDir, "runs", "detect", runName)
		a.training = trainingState{Running: true, Status: "Starting", Logs: []string{}, RunDir: runDir, Started: time.Now(), cancel: cancel}
		a.trainMu.Unlock()
		go a.runTraining(ctx, cancel, yoloBin, config, runName, runDir)
		writeJSON(w, map[string]any{"started": true, "run": runName})
	default: methodNotAllowed(w)
	}
}

func (a *App) validateTrainConfig(config *TrainConfig) error {
	config.Data = strings.TrimSpace(config.Data)
	config.Model = strings.TrimSpace(config.Model)
	config.Device = strings.TrimSpace(config.Device)
	if config.Data == "" { return errors.New("Dataset YAML path is required") }
	if !strings.EqualFold(filepath.Ext(config.Data), ".yaml") && !strings.EqualFold(filepath.Ext(config.Data), ".yml") { return errors.New("Dataset file must end in .yaml or .yml") }
	if !filepath.IsAbs(config.Data) { config.Data = filepath.Join(a.rootDir, config.Data) }
	if info, err := os.Stat(config.Data); err != nil || info.IsDir() { return errors.New("Dataset YAML file does not exist") }
	if config.Model == "" { config.Model = "yolov8n.pt" }
	if !modelPattern.MatchString(config.Model) { return errors.New("Model must be a .pt checkpoint path/name using only letters, numbers, /, _, ., :, or -") }
	if config.Epochs < 1 || config.Epochs > 10000 { return errors.New("Epochs must be between 1 and 10000") }
	if config.ImgSize < 32 || config.ImgSize > 2048 { return errors.New("Image size must be between 32 and 2048") }
	if config.Batch < -1 || config.Batch > 1024 || config.Batch == 0 { return errors.New("Batch must be -1 or between 1 and 1024") }
	if config.Device == "" { config.Device = "0" }
	if !devicePattern.MatchString(config.Device) { return errors.New("Device must be cpu, mps, or a GPU ID such as 0 or 0,1") }
	if config.Patience < 0 || config.Patience > 10000 { return errors.New("Patience must be between 0 and 10000") }
	if config.Workers < 0 || config.Workers > 64 { return errors.New("Workers must be between 0 and 64") }
	return nil
}

func (a *App) runTraining(ctx context.Context, cancel context.CancelFunc, yoloBin string, config TrainConfig, runName, runDir string) {
	defer cancel()
	args := []string{"detect", "train", "model=" + config.Model, "data=" + config.Data,
		fmt.Sprintf("epochs=%d", config.Epochs), fmt.Sprintf("imgsz=%d", config.ImgSize),
		fmt.Sprintf("batch=%d", config.Batch), "device=" + config.Device,
		fmt.Sprintf("patience=%d", config.Patience), fmt.Sprintf("workers=%d", config.Workers),
		"project=" + filepath.Join(a.rootDir, "runs", "detect"), "name=" + runName, "exist_ok=True"}
	cmd := exec.CommandContext(ctx, yoloBin, args...)
	cmd.Dir = a.rootDir
	stdout, err := cmd.StdoutPipe()
	if err != nil { a.finishTraining(err); return }
	stderr, err := cmd.StderrPipe()
	if err != nil { a.finishTraining(err); return }
	if err := cmd.Start(); err != nil { a.finishTraining(err); return }
	a.appendLog("$ " + yoloBin + " " + strings.Join(args, " "))
	a.setTrainingStatus("Training")
	var readers sync.WaitGroup
	readers.Add(2)
	read := func(pipe io.Reader) {
		defer readers.Done()
		scanner := bufio.NewScanner(pipe)
		scanner.Buffer(make([]byte, 4096), 1024*1024)
		for scanner.Scan() { a.appendLog(scanner.Text()) }
		if err := scanner.Err(); err != nil { a.appendLog("Output read error: " + err.Error()) }
	}
	go read(stdout)
	go read(stderr)
	err = cmd.Wait()
	readers.Wait()
	if err != nil {
		if ctx.Err() != nil { a.finishTraining(nil) } else { a.finishTraining(err) }
		return
	}
	a.trainMu.Lock()
	a.training.Running = false
	a.training.Status = "Completed"
	a.training.Finished = time.Now()
	a.training.cancel = nil
	a.trainMu.Unlock()
	a.appendLog("Training completed. Weights: " + filepath.Join(runDir, "weights", "best.pt"))
}

func (a *App) finishTraining(err error) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	a.training.Running = false
	a.training.Finished = time.Now()
	a.training.cancel = nil
	if err != nil { a.training.Status = "Failed"; a.training.Error = err.Error(); a.training.Logs = append(a.training.Logs, "Training failed: "+err.Error()) } else { a.training.Status = "Stopped"; a.training.Logs = append(a.training.Logs, "Training stopped.") }
}

func (a *App) setTrainingStatus(status string) { a.trainMu.Lock(); a.training.Status = status; a.trainMu.Unlock() }

func (a *App) appendLog(line string) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	a.training.Logs = append(a.training.Logs, line)
	if len(a.training.Logs) > 500 { a.training.Logs = append([]string(nil), a.training.Logs[len(a.training.Logs)-500:]...) }
}

func (a *App) handleStopTraining(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { methodNotAllowed(w); return }
	a.trainMu.Lock()
	if !a.training.Running || a.training.cancel == nil { a.trainMu.Unlock(); http.Error(w, "No training run is active", http.StatusConflict); return }
	a.training.Status = "Stopping"
	cancel := a.training.cancel
	a.trainMu.Unlock()
	cancel()
	writeJSON(w, map[string]bool{"stopping": true})
}

func (a *App) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet { methodNotAllowed(w); return }
	a.trainMu.Lock(); runDir := a.training.RunDir; a.trainMu.Unlock()
	if runDir == "" { writeJSON(w, map[string]any{"columns": []string{}, "rows": []MetricPoint{}}); return }
	metrics, err := readMetrics(filepath.Join(runDir, "results.csv"))
	if errors.Is(err, os.ErrNotExist) { writeJSON(w, map[string]any{"columns": []string{}, "rows": []MetricPoint{}}); return }
	if err != nil { http.Error(w, "Could not read training metrics: "+err.Error(), http.StatusInternalServerError); return }
	writeJSON(w, metrics)
}

type metricResponse struct { Columns []string `json:"columns"`; Rows []MetricPoint `json:"rows"` }

func readMetrics(path string) (metricResponse, error) {
	file, err := os.Open(path); if err != nil { return metricResponse{}, err }; defer file.Close()
	reader := csv.NewReader(file); reader.TrimLeadingSpace = true
	headers, err := reader.Read(); if err != nil { return metricResponse{}, err }
	response := metricResponse{Columns: headers, Rows: []MetricPoint{}}
	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) { break }
		if err != nil { return metricResponse{}, err }
		point := MetricPoint{}
		for i, value := range record {
			if i >= len(headers) { break }
			if number, err := strconv.ParseFloat(strings.TrimSpace(value), 64); err == nil && !math.IsNaN(number) && !math.IsInf(number, 0) { point[strings.TrimSpace(headers[i])] = number }
		}
		if len(point) > 0 { response.Rows = append(response.Rows, point) }
	}
	return response, nil
}

func (a *App) readLabels(imageName string) ([]Label, error) {
	labelName := strings.TrimSuffix(imageName, filepath.Ext(imageName)) + ".txt"
	contents, err := os.ReadFile(filepath.Join(a.labelsDir, labelName))
	if errors.Is(err, os.ErrNotExist) { return []Label{}, nil }; if err != nil { return nil, err }
	labels := make([]Label, 0)
	for lineNumber, line := range strings.Split(string(contents), "\n") {
		fields := strings.Fields(line); if len(fields) == 0 { continue }
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
	for _, label := range labels { fmt.Fprintf(&content, "%d %.6f %.6f %.6f %.6f\n", label.Class, label.X, label.Y, label.W, label.H) }
	return os.WriteFile(filepath.Join(a.labelsDir, labelName), []byte(content.String()), 0644)
}

func safeImageName(name string) (string, bool) {
	if name == "" || name != filepath.Base(name) || strings.ContainsAny(name, `/\\`) || !isImage(name) { return "", false }
	return name, true
}
func isImage(name string) bool { switch strings.ToLower(filepath.Ext(name)) { case ".jpg", ".jpeg", ".png", ".webp", ".bmp": return true; default: return false } }
func finite01(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1 }
func ensureJSONEnd(decoder *json.Decoder) error { var extra any; if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) { return errors.New("extra data") }; return nil }
func writeJSON(w http.ResponseWriter, value any) { w.Header().Set("Content-Type", "application/json; charset=utf-8"); _ = json.NewEncoder(w).Encode(value) }
func methodNotAllowed(w http.ResponseWriter) { w.Header().Set("Allow", "GET, POST, PUT"); http.Error(w, "Method not allowed", http.StatusMethodNotAllowed) }
