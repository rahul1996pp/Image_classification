import json
import os
import pickle
from random import choice
import shutil
import zipfile


class PKLToJSONConverter:
    """
    Converts `.pkl` files to `.json` files for a more structured and readable format.
    """
    @staticmethod
    def convert_to_json(pkl_file, json_file):
        """
        Convert a `.pkl` file to a `.json` file.

        Args:
            pkl_file (str): Path to the input `.pkl` file.
            json_file (str): Path to the output `.json` file.
        """
        try:
            # Load the pickle file
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)

            # Save the results as a JSON file
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Converted {pkl_file} to {json_file}")

        except Exception as e:
            print(f"Error converting {pkl_file}: {e}")

class ZipManager:
    """
    Optimized methods to zip and unzip files and directories with enhanced performance and cleaner handling.
    """

    @staticmethod
    def _zip_file(zipf, file_path, arcname):
        """Helper method to add a file to the zip file."""
        try:
            zipf.write(file_path, arcname)
        except Exception as e:
            print(f"Error zipping file {file_path}: {e}")

    @staticmethod
    def _zip_directory(zipf, directory_path, arcname=""):
        """
        Helper method to zip the contents of a directory.
        Args:
            zipf: The ZipFile object.
            directory_path: Path to the directory to be zipped.
            arcname: Archive name to retain folder structure.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Create relative path inside zip archive
                arcname_path = os.path.join(arcname, os.path.relpath(file_path, start=directory_path))
                ZipManager._zip_file(zipf, file_path, arcname_path)

    @staticmethod
    def zip_multiple_folders(folders, zip_file_path):
        """
        Compress multiple directories into a `.zip` file, preserving the directory structure.

        Args:
            folders (list): List of directories to be zipped.
            zip_file_path (str): Path to the output `.zip` file.
        """
        try:
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for folder in folders:
                    if os.path.isdir(folder):
                        print(f"Zipping directory: {folder}")
                        ZipManager._zip_directory(zipf, folder, os.path.basename(folder))
                    else:
                        print(f"Error: {folder} is not a valid directory.")
            print(f"Successfully zipped the directories into {zip_file_path}")
        except Exception as e:
            print(f"Error zipping folders: {e}")
    
    @staticmethod
    def unzip_file(zip_file_path, extract_dir):
        """
        Extracts a `.zip` file into a specified directory.
        
        Args:
            zip_file_path (str): Path to the `.zip` file to be extracted.
            extract_dir (str): Path to the directory where files will be extracted.
        """
        try:
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir, exist_ok=True)  # Create directory if it doesn't exist
            
            with zipfile.ZipFile(zip_file_path, 'r') as zipf:
                zipf.extractall(extract_dir)
            print(f"Successfully unzipped {zip_file_path} into {extract_dir}")
        except Exception as e:
            print(f"Error unzipping file {zip_file_path}: {e}")

class FileDeleter:
    """Class for deleting files and directories with error handling."""
    
    @staticmethod
    def delete(path):
        """Delete a file or directory (including its contents)."""
        try:
            if os.path.isfile(path):
                os.remove(path)  # Delete file
            elif os.path.isdir(path):
                shutil.rmtree(path)  # Delete directory and its contents
            print(f"{path} deleted successfully.")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

    @staticmethod
    def delete_directory_contents(directory_path):
        """Delete only the contents of a directory, not the directory itself."""
        try:
            if os.path.isdir(directory_path):
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    shutil.rmtree(item_path) if os.path.isdir(item_path) else os.remove(item_path)
                print(f"Contents of {directory_path} deleted.")
            else:
                print(f"Error: {directory_path} does not exist.")
        except Exception as e:
            print(f"Error deleting contents of {directory_path}: {e}")


class DetectionProcessor:
    def __init__(self):
        self.results = []
        self.all_files = set()
        self.file_to_hash = {}
        self.duplicates = set()
        self.unique_classifications = set()

    def process_json(self, input_data):
        """
        Processes the input JSON data and computes classifications, metadata, and results.

        Args:
            input_data (dict): Input JSON data to process.

        Returns:
            dict: Processed results and metadata.
        """
        for file_hash, data in input_data.items():
            files = data["files"]
            detections = data["detections"]
            classes = []
            # Track all files and detect duplicates
            for file in files:
                if file in self.file_to_hash:
                    self.duplicates.add(self.file_to_hash[file])
                    self.duplicates.add(file_hash)
                self.file_to_hash[file] = file_hash
            self.all_files.update(files)
            # Classification logic
            person_found = False
            class_counts = {}
            for detection in detections:
                cls = detection["Class"]
                if cls == "person":
                    person_found = True
                class_counts[cls] = class_counts.get(cls, 0) + 1
            if person_found:
                classification = "person"
            elif class_counts:
                # Find class with highest count, break ties randomly
                max_count = max(class_counts.values())
                most_common_classes = [cls for cls, count in class_counts.items() if count == max_count]
                classification = choice(most_common_classes)
            else:
                classification = "others"
            self.unique_classifications.add(classification)
            self.results.append({
                "hash": file_hash,
                "files": files,
                "class": list(class_counts.keys()),  # Unique classes
                "classification": classification
            })
        # Metadata
        metadata = {
            "all_files": list(self.all_files),
            "duplicates": list(self.duplicates),
            "folder_names": list(self.unique_classifications)  # Add folder names as unique classifications
        }
        return {"results": self.results, "metadata": metadata}


class JSONFileManager:
    @staticmethod
    def read_json(file_path):
        """
        Reads JSON data from a file.

        Args:
            file_path (str): Path to the input JSON file.

        Returns:
            dict: Parsed JSON data.
        """
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    @staticmethod
    def write_json(data, file_path):
        """
        Writes JSON data to a file.

        Args:
            data (dict): JSON data to write.
            file_path (str): Path to the output JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
            
            
class FolderManager:
    @staticmethod
    def create_folders(folder_list, base_path="."):
        """
        Creates folders from the given list at the specified base path.

        Args:
            folder_list (list): List of folder names to create.
            base_path (str): The base directory where folders should be created (default is current directory).
        """
        created_folders = []
        for folder_name in folder_list:
            folder_path = os.path.join(base_path, folder_name)
            try:
                os.makedirs(folder_path, exist_ok=True)
                created_folders.append(folder_path)
            except Exception as e:
                print(f"Error creating folder '{folder_name}': {e}")
        return created_folders
    
class FileMover:
    @staticmethod
    def move_files(file_list, target_folder):
        """
        Moves a list of files to the specified target folder.

        Args:
            file_list (list): List of file paths to move.
            target_folder (str): Path to the target folder where files will be moved.

        Returns:
            list: A list of successfully moved files.
        """
        moved_files = []
        os.makedirs(target_folder, exist_ok=True)  # Ensure the target folder exists

        for file in file_list:
            try:
                if os.path.isfile(file):  # Check if the file exists
                    target_path = os.path.join(target_folder, os.path.basename(file))
                    shutil.move(file, target_path)
                    moved_files.append(target_path)
                else:
                    print(f"File not found: {file}")
            except Exception as e:
                print(f"Error moving file '{file}': {e}")
        return moved_files

    @staticmethod
    def copy_files(file_list, target_folder):
        """
        Copies a list of files to the specified target folder.

        Args:
            file_list (list): List of file paths to copy.
            target_folder (str): Path to the target folder where files will be copied.

        Returns:
            list: A list of successfully copied files.
        """
        copied_files = []
        os.makedirs(target_folder, exist_ok=True)  # Ensure the target folder exists

        for file in file_list:
            try:
                if os.path.isfile(file):  # Check if the file exists
                    target_path = os.path.join(target_folder, os.path.basename(file))
                    shutil.copy2(file, target_path)  # Use copy2 to preserve metadata
                    copied_files.append(target_path)
                else:
                    print(f"File not found: {file}")
            except Exception as e:
                print(f"Error copying file '{file}': {e}")
        return copied_files
    
    @staticmethod
    def process_and_move_or_copy(processed_json, base_folder, operation="move"):
        """
        Processes the JSON output and moves or copies files to their respective folders based on classification.

        Args:
            processed_json (dict): The processed JSON containing file classifications and details.
            base_folder (str): The base folder where classified files will be organized.
            operation (str): Operation to perform - "move" or "copy". Defaults to "move".

        Returns:
            dict: A dictionary with results for each classification folder.
        """
        results = {}
        for item in processed_json.get("results", []):
            classification = item["classification"]
            files = item["files"]
            target_folder = os.path.join(base_folder, classification)
            if operation == "move":
                moved_files = FileMover.move_files(files, target_folder)
                results[classification] = {"operation": "moved", "files": moved_files}
            elif operation == "copy":
                copied_files = FileMover.copy_files(files, target_folder)
                results[classification] = {"operation": "copied", "files": copied_files}
            else:
                raise ValueError("Invalid operation. Use 'move' or 'copy'.")
        return results
