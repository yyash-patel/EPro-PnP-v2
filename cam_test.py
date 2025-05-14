import cv2
import os
import glob

# Set your image directory

# Get all image paths ending with _3d.jpg, sorted by name
image_paths = sorted(glob.glob(os.path.join("/simplstor/ypatel/workspace/EPro-PnP-v2/test", '*_3d.jpg')))

# Read the first image to get dimensions
frame = cv2.imread(image_paths[0])
height, width, _ = frame.shape

# Define video writer
out = cv2.VideoWriter('output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
                      1.0,                            # FPS
                      (width, height))                 # Frame size

# Write each frame
for path in image_paths:
    img = cv2.imread(path)
    # print(img.shape)
    if img is None:
        print(f"Warning: could not read {path}")
        continue
    out.write(img)

out.release()