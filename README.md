---

# Image Classification Using Object Detection Framework

This framework leverages **YOLO object detection** to classify images based on the objects detected within them. It automates the process of detecting objects in images, organizing them into categories, and saving results for further analysis. The system also supports file management, result caching, and batch processing to streamline workflows for large datasets.

---

## Key Features
- **Object Detection for Classification**: Uses YOLO to detect objects and classify images based on detected objects.
- **Batch Processing**: Processes multiple images in a folder, supports recursive directory traversal.
- **Caching**: Avoids redundant processing by caching detection results using file hashes.
- **Organized Output**: Classifies and organizes images into folders based on detected objects.
- **Results Export**: Saves detection results in `.pkl` and `.json` formats.
- **File Management**: Supports moving or copying images to classification-based folders.
- **Archiving**: Zips processed images and results for easy sharing or storage.

---

## How It Works
The framework processes images in a folder, detects objects using YOLO, and classifies images into categories based on the detected objects. For each image:
1. Objects are detected, and their bounding boxes, classes, and confidence scores are extracted.
2. Images are classified based on the detected objects.
3. Images are moved or copied into corresponding folders named after the object classes.
4. Results are saved in `.pkl` and `.json` formats for easy analysis.

---

## Requirements
- **Python**: 3.8 or higher.
- **YOLO**: Powered by the `ultralytics` library.
- **Additional Libraries**: `argparse`, `os`, `pickle`, `hashlib`.

Install dependencies using:
```bash
pip install ultralytics
```

---

## Setup
1. Place your images in the `img` directory (or specify a custom directory using `--image_dir`).
2. Results will be stored in the `results` directory by default (can be customized using `--results_dir`).

---

## Usage
The program provides a variety of operations controlled through command-line arguments.

### General Syntax
```bash
python script.py [OPTIONS]
```

### Key Options
| Option                  | Description                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------|
| `--image_dir`           | Input folder containing images (default: `img`).                                               |
| `--results_dir`         | Directory to store results (default: `results`).                                               |
| `--process_images`      | Detect objects in images and classify them.                                                    |
| `--pkltojson`           | Convert `.pkl` results to JSON format.                                                         |
| `--process_json`        | Organize files into folders based on detected objects.                                         |
| `--move_files`          | Move or copy files into classification folders.                                                |
| `--zip_process`         | Archive processed images and results into a zip file.                                          |
| `--delete_files`        | Delete the `results` directory after archiving.                                                |
| `--all`                 | Run all steps sequentially.                                                                    |

---

## Examples

### 1. Detect Objects and Classify Images
```bash
python script.py --process_images
```
Detects objects in images from `--image_dir` and saves results in `--results_dir`.

### 2. Convert Detection Results to JSON
```bash
python script.py --pkltojson
```
Converts detection results (`detections.pkl`) to JSON format (`detections.json`).

### 3. Organize Images into Folders by Classification
```bash
python script.py --process_json
```
Creates folders for each detected object class and moves images into these folders.

### 4. Move Classified Images
```bash
python script.py --move_files --operation move
```
Moves images into folders named after their object classes.

### 5. Archive Processed Files
```bash
python script.py --zip_process
```
Zips the `--image_dir` and `--results_dir` into a single file (`data.zip`).

### 6. Full Workflow
```bash
python script.py --all
```
Runs the entire workflow: detect objects, classify images, convert results to JSON, organize folders, archive, and clean up.

---

## Outputs
After running the program, you can expect the following outputs:
1. **Classified Folders**: Folders for each detected object class containing corresponding images.
2. **Detection Results**:
   - `detections.pkl`: Intermediate detection results in binary format.
   - `detections.json`: Results in JSON format for easy analysis.
   - `processed.json`: Organized JSON data linking files to classifications.
3. **Zip Archive**: Optional zipped folder containing processed files and results.

---

## Folder Structure

| File/Folder        | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `img/`             | Default directory for input images.                           |
| `results/`         | Directory for saving results (detections, JSON, etc.).        |
| `data.zip`         | Default zip archive for results.                              |
| `script.py`        | Main script for object detection and classification.          |
| `utils/`           | Helper modules for file handling, processing, and zipping.    |

---

## Customization
- **YOLO Model**: Replace `yolo11x.pt` with your own YOLO model in `ObjectDetector`.
- **File Extensions**: Extend `FileManager` to support additional image formats.
- **Classification Rules**: Modify `DetectionProcessor` for custom classification logic.

---

## Troubleshooting
1. **Model Loading Issues**:
   - Ensure the YOLO model file (`yolo11x.pt`) exists and is compatible with the `ultralytics` library.
   
2. **No Detections**:
   - Verify that the images are correctly formatted and contain detectable objects.

3. **Missing JSON/PKL**:
   - Ensure you run `--process_images` before running `--pkltojson` or `--process_json`.

---

**Buy me a coffee : [click here](https://www.paypal.me/RahulPujari "Pay")**

