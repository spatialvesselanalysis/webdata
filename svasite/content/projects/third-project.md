---
title: "Statistics"
date: 2025-05-21
weight: 3
---

# 3. Statistical Analyses
Now that you have quantitative data on your vessels, you can perform statistics to investigate what those numbers mean.

   - ## 3.1 Normalization
      Before you can apply statistical analyses, it is necessary to normalize your data. The code `Area_slide.py` (script downloadable from the user slrenne's erivessel repository) allows you to normalize the number of vessels on the tissue sample area. The function of this code is to segment the entire sample area through the L channel, also known as the luminance channel.

      To begin, set up the environment.
      ```python
      #python
      import cv2
      import numpy as np
      import matplotlib.pyplot as plt
      import os
      import glob
      import pandas as pd
      from tqdm import tqdm 
      ```
      The following function takes the path to an image as input and returns the area of tissue present, optionally saving visualizations of the results. The image is read using OpenCV, then converted from BGR to RGB to LAB (L = luminance, A = green/red, B = blue/yellow). Here, only the L channel is extracted to help separate tissue from the background based on brightness.
      ```python
      #python
      def calculate_tissue_area(image_path, save_visualizations=False, output_folder=None):
        
        img = cv2.imread(image_path)
        if img is None:
          print(f"Error: Could not read image {image_path}")
          return 0, None, None
    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
      ```
      You then apply a binary inverse threshold: pixels brighter than 220 (almost white) are set to 0 (black), while others are set to 255 (white). This isolates the darker tissue from the bright background. Next, you perform morhological operations and find the contours of white regions in the binary image, which should correspond to tissue.
      ```python
      #python
        _, binary = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY_INV)
    
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
        mask = np.zeros_like(l_channel)

        min_contour_area = 1000  # Adjust based on your image size
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(mask, [contour], 0, 255, -1)
      ```
      Here, you count how many white pixels are present, knowing that each pixel corresponds to 1 unit of tissue area. Additionally a mask is used to extract only the tissue from the RGB image by making pixels outside the tissue black.
      ```python
      #python
        area_pixels = cv2.countNonZero(mask)
        filled_tissue = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
      ```
      If enabled, the `save_visualizations` function generates a 2x2 subplot with: the original image, the binary threshold result, the tissue mask and the overlayed image with area shown in the title. The plots are then saved in the specified `output_folder` with `_analysis.png` suffix.
      ```python
        if save_visualizations:
            if output_folder is None:
                output_folder = os.path.dirname(image_path) or '.'
        
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.imshow(img_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(binary, cmap='gray')
            plt.title('Thresholded Image')
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(mask, cmap='gray')
            plt.title('Tissue Mask')
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(filled_tissue)
            plt.title(f'Filled Tissue (Area: {area_pixels} pixels)')
            plt.axis('off')
            
            plt.tight_layout()
        
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            plt.savefig(f"{output_folder}/{base_name}_analysis.png", dpi=300)
            plt.close()
      ```
      At the end the `calculate_tissue_area` function returns the tissue area in pixels, the binary mask, and the RGB image with tissue highlighted.
      ```python   
        return area_pixels, mask, filled_tissue
      ```
      Example output: a binary image showing the background in black and the total tissue area in white.
      <p align="center">
        <img src="/images/Area.png" alt="Binary image showing tissue as white and background as black" width="400">
      </p>

      The next function, `process_folder`, processes all image files in a given folder and applies the `calculate_tissue_area` function to each of them. The results are saved to a CSV file.
      ```python
      def process_folder(input_folder, output_folder, file_extensions=None):
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
      
        os.makedirs(output_folder, exist_ok=True)
        
        image_files = []
        for ext in file_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
        
        if not image_files:
            print(f"No image files found in {input_folder} with extensions {file_extensions}")
            return None
        
        print(f"Found {len(image_files)} image files to process")
        
        results = []
        
        for img_path in tqdm(image_files, desc="Processing slides"):
            try:
                filename = os.path.basename(img_path)
                area, mask, filled_tissue = calculate_tissue_area(img_path)
      
                img = cv2.imread(img_path)
                base_name = os.path.splitext(filename)[0]
                cv2.imwrite(f"{output_folder}/{base_name}_original.png", img)
                
                if mask is not None:
                    cv2.imwrite(f"{output_folder}/{base_name}_mask.png", mask)
                
                results.append({
                    'Image': filename,
                    'Tissue Area (pixels)': area
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_folder, "tissue_areas.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
            return df
        else:
            print("No results were generated")
            return None   
      ```
      The final section of this script is the main execution block. It defines the folders to use, calls the processing function and checks whether results were produced by printing a confirmation or warning message accordingly.
      ```python
      if __name__ == "__main__":
        input_folder = 'Input'
        output_folder = 'Output'
        
        print(f"Processing slides from {input_folder}")
        print(f"Results will be saved to {output_folder}")
        
        df = process_folder(input_folder, output_folder)
        
        if df is not None:
            print("Processing complete!")
            print(f"CSV file with area measurements saved to {output_folder}/tissue_areas.csv")
        else:
            print("No results were generated.")
      ```
      Now that you have calculated the area of each tissue section in pixels, you can proceed in performing your desired statistical analyses normalizing on tissue area. 
