import argparse
import os
import hashlib
import pickle
import sys
from ultralytics import YOLO
from utils import DetectionProcessor, FileDeleter, FileMover, FolderManager, JSONFileManager, PKLToJSONConverter, ZipManager


class CacheManager:
    """
    Handles caching of detection results to avoid redundant processing.
    """
    def __init__(self, cache_file='detections.pkl'):
        self.cache_file = cache_file

    def _load_cache(self):
        """Load the entire cache from file."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache):
        """Save the entire cache to file."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache, f)

    def get(self, key):
        """Retrieve a cached result by key."""
        cache = self._load_cache()
        return cache.get(key, None)

    def set(self, key, data):
        """Save a result to the cache."""
        cache = self._load_cache()
        cache[key] = {'data': data}  # Store data under the 'data' key
        self._save_cache(cache)


class FileManager:
    """
    Handles file and directory operations for input images and output results.
    """
    @staticmethod
    def generate_cache_key(file_path):
        """Generate a unique cache key based on file content."""
        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()

    @staticmethod
    def get_image_files(input_path, recursive=True):
        """Get a list of image files from a file or folder."""
        if os.path.isfile(input_path):
            return [input_path]
        file_list = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    file_list.append(os.path.join(root, file))
            if not recursive:
                break
        return file_list

    @staticmethod
    def save_results(output_path, results):
        """Save results to a `.pkl` file."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            temp_file = output_path + ".tmp"
            with open(temp_file, 'wb') as f:
                pickle.dump(results, f)
            os.rename(temp_file, output_path)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")

    @staticmethod
    def load_results(output_path):
        """Load results from a `.pkl` file."""
        try:
            with open(output_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading results from {output_path}: {e}")
            return {}


class ObjectDetector:
    """
    Handles object detection using YOLO and integrates caching and batch processing.
    """
    def __init__(self, model_path='yolo11x.pt', cache_file='detection_cache.pkl'):
        self.model = YOLO(model_path)
        self.cache_manager = CacheManager(cache_file)

    def detect_objects(self, image_path):
        """
        Detect objects in a single image.

        Args:
            image_path (str): Path to the image.

        Returns:
            dict: A dictionary containing the hash and detection results.
        """
        # Generate cache key (MD5 hash of the file content)
        cache_key = FileManager.generate_cache_key(image_path)

        # Check cache for the key
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            print(f"Loaded from cache: {image_path}")
            return {"hash": cache_key, "detections": cached_result['data']}

        # Perform detection if not found in cache
        results = self.model(image_path)
        detections = []
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                detections.append({
                    "Bounding Box": box.tolist(),
                    "Class": result.names[int(cls)],
                    "Confidence": float(conf)
                })

        # Cache the result along with the hash
        self.cache_manager.set(cache_key, detections)
        print(f"Computed and cached: {image_path}")
        return {"hash": cache_key, "detections": detections}

    def process_batch(self, input_path, output_path, recursive=True):
        """
        Process a batch of images and save results in a `.pkl` file.

        Args:
            input_path (str): Path to an image file or folder.
            output_path (str): Path to save results.
            recursive (bool): Whether to process folders recursively.

        Returns:
            dict: A dictionary of image paths and detection results.
        """
        file_list = FileManager.get_image_files(input_path, recursive)
        results = {}

        for file_path in file_list:
            results[file_path] = self.detect_objects(file_path)

        FileManager.save_results(output_path, results)
        return results


class BatchProcessor:
    """
    Handles batch processing of images, checking for unprocessed files,
    detecting objects, and saving results. Also manages file renaming and moving.
    """
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def get_unprocessed_files(self):
        """
        Retrieve all image files and check for unprocessed files.
        This includes handling renamed or moved files with the same hash.
        """
        all_files = FileManager.get_image_files(self.input_path, recursive=True)
        loaded_results = FileManager.load_results(self.output_path)
        hash_to_data = loaded_results  # Hashes are the keys, and values are file data and detections

        unprocessed_files = []
        
        for file in all_files:
            file_hash = FileManager.generate_cache_key(file)

            if file_hash in hash_to_data:
                # If file already processed, check if the old file exists
                existing_files = hash_to_data[file_hash]["files"]

                # Check if the old file exists and if the new file exists
                if not any(os.path.exists(old_file) for old_file in existing_files):
                    # The old file doesn't exist but new file exists, replace old with new
                    if os.path.exists(file) and file not in existing_files:
                        hash_to_data[file_hash]["files"] = [file]
                else:
                    # If old file exists, just add the new file to the list if not already present
                    if os.path.exists(file) and file not in existing_files:
                        existing_files.append(file)
            else:
                # This is a new file (not in the results yet)
                if os.path.exists(file):
                    hash_to_data[file_hash] = {
                        "files": [file],  # Start with the file in the list
                        "detections": []  # Empty detections for now; will be updated after detection
                    }
                    unprocessed_files.append(file)

        # Now, ensure no duplicates in the "files" list
        for file_hash, data in hash_to_data.items():
            data["files"] = list(set(data["files"]))  # Remove duplicates by converting to set and back to list

        return hash_to_data, unprocessed_files

    def process_unprocessed_files(self, unprocessed_files, hash_to_data, cache_file):
        """
        Process the unprocessed files and update the results.
        Only new files will be processed.
        """
        if unprocessed_files:
            self.detector = ObjectDetector(cache_file=cache_file)
            print(f"Processing {len(unprocessed_files)} unprocessed files...")
            for file_path in unprocessed_files:
                if os.path.exists(file_path):
                    result = self.detector.detect_objects(file_path)
                    file_hash = result["hash"]
                    detections = result["detections"]
                    # Only update detections for the first occurrence of this hash
                    if not hash_to_data[file_hash]["detections"]:
                        hash_to_data[file_hash]["detections"] = detections
                    # Add the file to the list of files under this hash, if not already present
                    if file_path not in hash_to_data[file_hash]["files"]:
                        hash_to_data[file_hash]["files"].append(file_path)

        else:
            print("All files are already processed.")

        # Save updated results including filename changes and duplicates
        FileManager.save_results(self.output_path, hash_to_data)

    def display_results(self, hash_to_data):
        """
        Display the results of the object detection, including duplicates.
        """
        for file_hash, data in hash_to_data.items():
            print(f"Results for hash: {file_hash}")
            print(f"  Files: {data['files']}")
            for detection in data["detections"]:
                print(f"    {detection}")


def main():

    # Setting up argument parsing
    parser = argparse.ArgumentParser(description="Process image detections and organize results.")
    parser.add_argument("--image_dir", type=str, default="img", help="Input folder containing images.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to store intermediate and final results.")
    parser.add_argument("--zip_file", type=str, default="data.zip", help="Path to the output zip file.")
    parser.add_argument("--operation", type=str, choices=["move", "copy"], default="move",
                        help="Operation to perform on processed files: 'move' or 'copy'.")
    # Flags to control specific steps
    parser.add_argument("--process_json", action="store_true", help="Process JSON from detections.")
    parser.add_argument("--process_images", action="store_true", help="Process images for detections.")
    parser.add_argument("--pkltojson", action="store_true", help="Convert PKL to JSON.")
    parser.add_argument("--move_files", action="store_true", help="Move or copy files based on classification.")
    parser.add_argument("--zip_process", action="store_true", help="Zip processed files and results.")
    parser.add_argument("--delete_files", action="store_true", help="Delete results directory after zipping.")
    parser.add_argument("--all", action="store_true", help="Run all steps in sequence.")

    args = parser.parse_args()

    # If --all is set, enable all individual flags
    if args.all:
        args.process_images = True
        args.pkltojson = True
        args.process_json = True
        args.move_files = True
        args.zip_process = True
        args.delete_files = True

    # Using the arguments
    image_dir = args.image_dir
    results_dir = args.results_dir
    zip_file = args.zip_file
    classify_images_dir = os.path.join(results_dir, "processed_images")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(classify_images_dir, exist_ok=True)
    pkl_file = os.path.join(results_dir, "detections.pkl")  # Path to save detection results
    json_file = os.path.join(results_dir, "detections.json")  # Path to the output .json file
    processed_json_file = os.path.join(results_dir, "processed.json")

    # Dependency checks
    if args.process_json and not os.path.exists(json_file):
        print("Error: JSON file does not exist. Please run with --pkltojson first.")
        sys.exit(1)

    if args.move_files and not os.path.exists(processed_json_file):
        print("Error: Processed JSON file does not exist. Please run with --process_json first.")
        sys.exit(1)

    if args.zip_process and not (os.path.exists(image_dir) or os.path.exists(results_dir)):
        print("Error: Nothing to zip. Ensure images and results are available.")
        sys.exit(1)

    # Conditional processing based on flags
    if args.process_images:
        print("Processing images...")
        processor = BatchProcessor(image_dir, pkl_file)
        # Get unprocessed files
        loaded_results, unprocessed_files = processor.get_unprocessed_files()
        # Process unprocessed files
        processor.process_unprocessed_files(unprocessed_files, loaded_results, pkl_file)
        # Display results
        processor.display_results(loaded_results)

    if args.pkltojson:
        print("Converting PKL to JSON...")
        if not os.path.exists(pkl_file):
            print(f"Error: PKL file '{pkl_file}' not found. Please process images first with --process_images.")
            sys.exit(1)
        PKLToJSONConverter.convert_to_json(pkl_file, json_file)

    if args.process_json:
        print("Processing JSON...")
        dp = DetectionProcessor()
        processed_json = dp.process_json(JSONFileManager.read_json(json_file))
        JSONFileManager.write_json(processed_json, processed_json_file)
        FolderManager.create_folders(processed_json["metadata"]["folder_names"], classify_images_dir)

    if args.move_files:
        print(f"{args.operation.capitalize()} files based on classification...")
        processed_json = JSONFileManager.read_json(processed_json_file)
        FileMover.process_and_move_or_copy(processed_json, classify_images_dir, operation=args.operation)

    if args.zip_process:
        print("Zipping processed files...")
        ZipManager.zip_multiple_folders([image_dir, results_dir], zip_file)

    if args.delete_files:
        print("Deleting results directory...")
        if os.path.exists(results_dir):
            FileDeleter.delete(results_dir)
        else:
            print("Error: Results directory does not exist, nothing to delete.")


if __name__ == "__main__":
    main()
