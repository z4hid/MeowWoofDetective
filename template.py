import os
from pathlib import Path
import logging

# Step 00: Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)]: %(message)s: ')  # Log messages with ASCII time

# Step 01: Define the project name
project_name = "cnnClassifier"

# Step 02: Define a list of files to create
list_of_files = [
    ".github/workflows/ .gitkeep",  # Keep the .github/workflows directory
    f"src/{project_name}/__init__.py",  # Create __init__.py in src/cnnClassifier directory
    f"src/{project_name}/components/__init__.py",  # Create __init__.py in src/cnnClassifier/components directory
    f"src/{project_name}/utils/__init__.py",  # Create __init__.py in src/cnnClassifier/utils directory
    f"src/{project_name}/config/__init__.py",  # Create __init__.py in src/cnnClassifier/config directory
    f"src/{project_name}/pipeline/__init__.py",  # Create __init__.py in src/cnnClassifier/pipeline directory
    f"src/{project_name}/entity/__init__.py",  # Create __init__.py in src/cnnClassifier/entity directory
    f"src/{project_name}/constants/__init__.py",  # Create __init__.py in src/cnnClassifier/constants directory
    "config/config.yaml",  # Create config.yaml file
    "params.yaml",  # Create params.yaml file
    "requirements.txt",  # Create requirements.txt file
    "setup.py",  # Create setup.py file
    "research/trials.ipynb"  # Create trials.ipynb file in the research directory
]

# Step 03: Iterate over each file path
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert file path to a Path object
    filedir, filename = os.path.split(filepath)  # Get the file directory and filename

    # Step 04: Create the file directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # Create directory if it doesn't exist
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    # Step 05: Create an empty file if it doesn't exist or has size 0
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")

    # Step 06: Log if the file already exists
    else:
        logging.info(f"{filename} already exists!")
