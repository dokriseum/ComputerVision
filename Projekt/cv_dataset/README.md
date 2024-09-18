# Computer Vision Project

This project manages and converts datasets for a **Computer Vision** SoSe2024 semester project. The datasets consist of various image categories, which are processed to maintain a consistent naming convention and format.

## Dataset Overview

The datasets are categorized into three main groups:

1. **Bike**: Contains images related to bicycles.
2. **Bin**: Contains images of trash bins.
3. **Shield**: Contains images of traffic signs displaying street names from Berlin. This dataset is specifically designed for image recognition of street names in the Berlin area.

### Dataset Links on HTW-Cloud

download the datasets from the HTW Cloud using the following link:

- [CV_Project_Dataset](https://cloud.htw-berlin.de/f/148192704)

## Image Naming and Conversion

The provided datasets include images in various formats such as `.jpg`, `.jpeg`, `.bmp`, and `.png`. All images are converted into **`.png`** format for uniformity. The images are also renamed according to a structured naming convention based on their respective categories:

- **Bike**: `bike001.png`, `bike002.png`, ...
- **Bin**: `bin001.png`, `bin002.png`, ...
- **Shield**: `shield001.png`, `shield002.png`, ...

For every new image, the numbering continues from the last available image to avoid overwriting existing files.

### Conversion Script

The `main.py` script is responsible for converting and renaming the image files. It performs two main functions:

1. Converts images to `.png` format.
2. Renames the images based on a sequential naming scheme starting from the next available number in the category.

#### Key Instructions for Using the Script:

- **Folder Paths**: Ensure the `source_folder` and `destination_folder` paths in `main.py` are correct for your system.
- **Image Numbering**: The script automatically determines the next number for naming new images by checking the existing files in the destination folder.

### How to Run the Script

Follow these steps to use the script for converting and renaming images:

1. Download the dataset files.
2. Open `main.py` and adjust the paths for `source_folder` and `destination_folder` based on your setup.
3. Open a terminal or command prompt, navigate to the project directory, and run the script:

```bash
python main.py
```
