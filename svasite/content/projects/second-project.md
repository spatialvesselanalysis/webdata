---
title: "Vessel Segmentation"
date: 2025-05-21
weight: 2
---

# 2. Segment Vessels and Extract Features in Python
This is where the true transformation — or as some may say, magic — happens. With the images at hand, you can perform vessel segmentation and feature extraction using Python. Simply run the `Segmentation.py` (script downloadable from the user slrenne's erivessel repository) script as follows.

In the prelinimary setup, upload necessary libraries.
```python
#python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import measure
import pandas as pd
import os
```

Next, define the path that sets the directory where the images are stored and create a list of subdirectories (folders) within the specified path.
```python
#python
PATH = "Insert/Path/to/Folders"

folders = [item for item in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, item))]
```

Now, loop through each folder to process the images contained within, construct the image path, and read the image using OpenCV. Additionally, you can check if a directory for saving segmented images exists, and create it if not.
```python
#python
for folder in folders:
    general_path = os.path.join(PATH, folder)
    image_path = general_path + "/" + folder + ".jpg"
    original = cv2.imread(image_path)

    print(f"Image - {image_path} is read")

    if not os.path.exists(f"{general_path}/{folder}_SEGMENTATION"):
        os.makedirs(f"{general_path}/{folder}_SEGMENTATION")

    directory_to_save = general_path + "/" + folder + "_SEGMENTATION/"  
```

   - ## 2.1 Convertion of RGB to HSV format
      Blood vessels love to hide, but their secrets can be revealed with the right tips and tricks. The first trick is converting the images from the standard RGB (Red, Green, Blue) colour space to HSV (Hue, Saturation, Value) format. This transformation is particularly useful because the H channel enhances the contrast of blood vessels, making them easier to distinguish from the surrounding tissue. Using the OpenCV library, the conversion is straightforward
      ```python
      #python
      h_s_v_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
      h, s, v = cv2.split(h_s_v_image)
      h = 255 - h
      ```
   
   - ## 2.2 Application of Otsu’s thresholding
      Once the H channel is extracted, Otsu’s thresholding is applied to create a binary mask. This technique automatically determines an optimal threshold, separating vessel structures from the background. 
      ```python
      #python
      ret, binary = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
      binary = binary.astype(np.uint8)
      binary *= 255
      ```
      
   - ## 2.3 Morphological operations
      You fine-tune the segmentation by cleaning up noise and ensuring vessel structures are continuous thanks to morphological operations. Feel free to play with the parameters below to tailor the output to your necessities, according to the characteristics of your histological images and the level of detail required for your analysis.
      - If vessels appear fragmented → Increase dilation iterations or use a larger segmentation kernel.
      - If vessels merge too much → Decrease dilation iterations or use a smaller segmentation kernel.
      - If there are small gaps in vessels → Increase closing iterations or kernel size.
      - If fine details are lost → Use a smaller closing kernel or fewer iterations.
   
      ```python
      #python
      kernel_for_structuring_element_for_segmentation = (15, 15)
      kernel_for_structuring_element_for_closing = (15, 15)
      iterations_for_dilate = 4
      iterations_for_closing = 3
       ```
      Dilation is used to bridge small gaps between fragmented vessel structures, improving connectivity. Closing, which involves dilation followed by erosion, eliminates small holes within vessels. Additionally, you may find the boundaries of the detected vessels to fill them in the binary image.
      ```python
      #python
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_for_structuring_element_for_segmentation)
      binary = cv2.dilate(binary, kernel, iterations=iterations_for_dilate)
   
      cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
   
      for c in cnts:
         cv2.drawContours(binary, [c], 0, (255, 255, 255), -1)
      
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_for_structuring_element_for_closing)
      opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations_for_closing)
      ```

      Now that you have finalized the segmentation, save the processed image to the specified directory.
      ```python
      #python
      plt.imsave(f"{directory_to_save}{folder}_Segmented.png", opening, cmap="gray")
      ```
      Example output: a binary image, created using a mask, where all vessels are represented in white and the background is shown in black.

      <p align="center">
        <img src="/images/Segmented.png" alt="Image of segmented vessels" width="400">
      </p>

      To enhance interpretability - because seeing is believing - contour detection outlines vessel boundaries as an overlay on the original images. Nothing beats a well-labeled, highlighted image that speaks for itself, showcasing exactly what you've extracted.
      ```python
      #python
      contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
      for c in contours:
         cv2.drawContours(original, [c], 0, (0, 255, 0), 5)
          
       cv2.imwrite(f"{directory_to_save}{folder}_Vessels.jpg", original)
      ```
      Example output: the original histological image, with vessel countours highlighted in green.
      <p align="center">
        <img src="/images/Contoured.jpg" alt="Image of contoured vessels" width="400">
      </p>

   - ## 2.4 Feature extraction
     Great, you now have segmented vessels, but science demands numbers, not just pretty pictures. To obtain more information on vessel morphology, quantitative features are extracted using `regionprops_table` from `skimage.measure`. These properties include:
      - `area`: The number of pixels inside each segmented vessel.
      - `axis_major_length`: The length of the longest axis of the vessel (major axis of the fitted ellipse).
      - `axis_minor_length`: The length of the shortest axis of the vessel (minor axis of the fitted ellipse).
      - `eccentricity`: Measures how close to a circle the vessel is (0 = perfect circle, between 0 and 1 = ellipse, 1 = parabola).
      - `orientation`: The angle in degrees of the major axis relative to the horizontal axis.
   
     The `label()` function from `skimage.measure` assigns a unique integer label to each connected component - in our case a segmented vessel - in the binary image, this helps in identifying individual vessels. You then define the list of desired morphological properties and compute them.
     ```python
     #python
     label_img = label(opening)
     all_props = ["area", "axis_major_length", "axis_minor_length", "eccentricity", "orientation"]
     props = regionprops(label_img)
     props = measure.regionprops_table(label_img, original, properties=all_props)
     data = pd.DataFrame(props)
     data.to_csv(f'{directory_to_save}Measurements_{folder}.csv', index=True)
     ```
      The output is a CSV file `{folder}_Measurements.csv` containing quantitative measurements of each detected vessel in the segmented image. The columns of the CSV file represent different morphological properties of the vessels, and each row corresponds to one labeled vessel.