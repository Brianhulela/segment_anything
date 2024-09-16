from ultralytics import SAM
import urllib.request
import supervision as sv
import cv2
import matplotlib.pyplot as plt

# Download and read image
url, filename = ("https://images.unsplash.com/photo-1532186232057-80e418ed6614?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTIwfHxzdHJlZXQlMjB3aXRoJTIwY2FycyUyMGFuZCUyMHBlb3BsZXxlbnwwfHwwfHx8MA%3D%3D", "scene.jpg")
urllib.request.urlretrieve(url, filename)    # Download image
image_rgb = cv2.imread(filename)

# Load the SAM model
model = SAM("sam2_t.pt")

# Get segmentation results
results = model(filename)

for i, result in enumerate(results):  # Iterate over each result in the list
    print(f"Processing result {i+1}/{len(results)}")
            
    # Check if the current result has masks
    if result.masks is not None:
        masks = result.masks.data  # Access the mask data
                
        boxes = result.boxes.xyxy  # Get the bounding box coordinates
        print(f"Mask shape for result {i+1}: {masks.shape}")
                
        # Convert the PyTorch tensor boxes to a NumPy array
        boxes_np = boxes.cpu().numpy()  # Convert boxes to NumPy format
        masks_np = masks.cpu().numpy()  # Convert masks to NumPy format
                
        # If masks exist, apply them to the original image
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
                
        # Create detections manually
        detections = sv.Detections(xyxy=boxes_np, mask=masks_np)
                
        # Annotate the image
        annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        
        sv.plot_images_grid(
            images=[image_rgb, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )
        
        # Save the annotated image locally
        cv2.imwrite("segmented_image.jpg", annotated_image)  # Convert back to BGR for saving
        
    else:
        print(f"No masks found in result {i+1}")
