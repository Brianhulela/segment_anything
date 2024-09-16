# SAM Image Segmentation

This project demonstrates how to use the Segment Anything Model (SAM) from the `ultralytics` library for image segmentation. The script downloads an image, applies SAM to perform segmentation, and saves the segmented image locally.

## Features
- Download an image from a URL.
- Perform image segmentation using the SAM model.
- Annotate and visualize the segmentation results.
- Save the segmented image locally.

## Requirements

- Python 3.x
- `ultralytics` (for the SAM model)
- `supervision` (for handling detection annotations)
- `opencv-python` (for image reading, saving, and processing)

### Install dependencies

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

### How It Works
The script performs the following steps:

1. Downloads an image from the specified URL.
2.  Loads the SAM model using ultralytics.
3. Processes the image and performs segmentation using the SAM model.
4. Annotates the detected segments (masks and bounding boxes) on the original image.

Saves the segmented image locally.
