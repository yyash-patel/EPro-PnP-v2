# import cv2
# import numpy as np
# import os

# def stitch_3d_images(folder1, folder2, output_folder):
#     os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

#     # Get list of 3D images only (filenames ending with '_3d.jpg')
#     images1 = sorted([f for f in os.listdir(folder1) if f.endswith('_3d.jpg')])
#     images2 = sorted([f for f in os.listdir(folder2) if f.endswith('_3d.jpg')])

#     # Find common 3D filenames
#     common_files = set(images1) & set(images2)

#     for filename in common_files:
#         img1_path = os.path.join(folder1, filename)
#         img2_path = os.path.join(folder2, filename)

#         img1 = cv2.imread(img1_path)
#         img2 = cv2.imread(img2_path)

#         if img1 is None or img2 is None:
#             print(f"Skipping {filename}: Could not read one of the images.")
#             continue

#         # Resize images to have the same height
#         min_height = min(img1.shape[0], img2.shape[0])
#         img1 = cv2.resize(img1, (int(img1.shape[1] * min_height / img1.shape[0]), min_height))
#         img2 = cv2.resize(img2, (int(img2.shape[1] * min_height / img2.shape[0]), min_height))

#         # Stitch images side by side
#         stitched_img = np.hstack((img1, img2))

#         # Save the result
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, stitched_img)
#         print(f"Saved: {output_path}")

#     print("All 3D images processed!")

# # Example usage
# folder1 = "/simplstor/ypatel/workspace/single-image-pose/data/frames/fine_tuned"
# folder2 = "/simplstor/ypatel/workspace/single-image-pose/data/frames/original"
# output_folder = "stitched_output"

# stitch_3d_images(folder1, folder2, output_folder)


import cv2
import numpy as np
import os

def stitch_3d_images_with_labels(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

    # Get list of 3D images only (filenames ending with '_3d.jpg')
    images1 = sorted([f for f in os.listdir(folder1) if f.endswith('_3d.jpg')])
    images2 = sorted([f for f in os.listdir(folder2) if f.endswith('_3d.jpg')])

    # Find common 3D filenames
    common_files = set(images1) & set(images2)

    for filename in common_files:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Skipping {filename}: Could not read one of the images.")
            continue

        # Resize images to have the same height
        min_height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * min_height / img1.shape[0]), min_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * min_height / img2.shape[0]), min_height))

        # Add labels to images
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0)  # White text
        thickness = 2
        line_type = cv2.LINE_AA

        # Add "Original" label to left image
        cv2.putText(img1, "FINE-TUNED MODEL", (10, 30), font, font_scale, font_color, thickness, line_type)

        # Add "Fine-Tuned" label to right image
        cv2.putText(img2, "ORIGINAL MODEL", (10, 30), font, font_scale, font_color, thickness, line_type)

        # Stitch images side by side
        stitched_img = np.hstack((img1, img2))

        # Save the result
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, stitched_img)
        print(f"Saved: {output_path}")

    print("All 3D images processed!")

# Example usage
folder1 = "/simplstor/ypatel/workspace/single-image-pose/data/real_test_data/viz"
folder2 = "/simplstor/ypatel/workspace/single-image-pose/data/real_test_data/viz_original"
output_folder = "stitched_output"

stitch_3d_images_with_labels(folder1, folder2, output_folder)