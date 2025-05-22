---
title: "Data Preparation"
date: 2025-05-22
weight: 1
---

# 1. Prepare Your Data
Before diving into analysis, you need to prepare your images. Here’s what you need to get your data ready for processing:

   - ## 1.1 Stain the histological images of interest for CD31
     The first step is to obtain the histological images of interest. These images must include a stain that highlights blood vessels. Common choices are CD31 or CD34, both of which label the endothelial cells lining the vasculature. CD31 is particularly useful due to its strong and specific endothelial staining, while CD34 provides additional coverage of endothelial and progenitor cells. These stains are essential for enhancing vessel contrast and enabling analyses further down the pipeline. Finally, the slides need to be scanned digitally for further evaluation.

   - ## 1.2 Ensure the images are formatted correctly
     Once you have your images, it’s crucial that they are in the right format for compatibility with the analysis pipeline. The most commonly used image formats for this type of work are `.tiff` (higher quality, slower processing) and `.jpg` (lower quality, faster processing).
     
   - ## 1.3 Remove image artifacts
     Unfortunately, microscope WSIs often suffer from artifacts. Being able to remove them is a crucial step to ensure that downstream analyses are not misled by irrelevant visual noise. The Python script `ArtifactRemover.py` (script downloadable from the user slrenne's erivessel repository) allows you to manually or automatically exclude artifacts from WSIs. It provides both a graphical interface and interactive image filtering options to clean up unwanted areas of tissue from further processing. You can choose between freehand cropping or automatic thresholding based on how many images you have: cropping is ideal for a few images where precision is needed, while automatic L-channel filtering is better suited for batch processing many slides quickly.

     To begin, set up your environment by loading all necessary libraries. If they are not already downloaded to your computer, install them through the `pip` function.
     ```python
     #python
     import cv2
     import cv2
     import numpy as np
     import matplotlib
     matplotlib.use('TkAgg')
     import matplotlib.pyplot as plt
     from skimage import color
     from matplotlib.widgets import Slider
     from tkinter import Tk, Button, filedialog, messagebox, Label
     import os
     from PIL import Image
     import tempfile
     ```
     The class `WSIProcessor` encapsulates all functionality to load an image, crop it manually or filter it automatically, and save the output. It takes a single image path as input and stores both the original and processed versions of the image. It reads the image using OpenCV in BGR format and it converts it to RGB so it displays correctly with matplotlib and PIL.
     ```python
     #python
     class WSIProcessor:
      def __init__(self, image_path):
          self.image_path = image_path
          self.original_image = cv2.imread(image_path)
          if self.original_image is None:
              raise FileNotFoundError(f"Image not found: {image_path}")
          self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
          self.cropped_image = None
     ```
     The function `show_image` displays any image in a blocking matplotlib window.
     ```python
     #python
      def show_image(self, img, title='Image', cmap=None):
          plt.figure(figsize=(10, 10))
          plt.imshow(img, cmap=cmap)
          plt.title(title)
          plt.axis('off')
          plt.show(block=True) 
     ```
     The function `save_result` uses a file dialog to let you choose where to save the processed image. it converts the NumPy array to a PIL Image and saves it as default in `.png` format.
     ```python
     #python
      def save_result(self, img):
          save_path = filedialog.asksaveasfilename(
              defaultextension=".png",
              filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
              initialfile="processed_image.png"
          )
          if save_path:
              Image.fromarray(img).save(save_path)
              messagebox.showinfo("Saved", f"Image saved to:\n{save_path}")
     ```
     After the basic functions to open and save images, it's time to start processing your images. The `interactive_crop` function implements freehand region drawing using OpenCV's `cv2.setMouseCallback`. Here you draw points on the image(s) with your mouse to isolate the tissue areas you wish to keep and then press ENTER to finalize the outlined polygon from which a binary mask is created. The image is masked and pasted on a white background. The result is a cropped image that excludes any artifacts or unwanted areas from further processing.
     ```python
     #python
      def interactive_crop(self):
          print(f"Draw freehand region on: {os.path.basename(self.image_path)}")
          clone = self.image.copy()
          points = []

          def draw_polygon(event, x, y, flags, param):
              if event == cv2.EVENT_LBUTTONDOWN:
                  points.append((x, y))
                  cv2.circle(clone, (x, y), 2, (0, 255, 0), -1)
                  if len(points) > 1:
                      cv2.line(clone, points[-2], points[-1], (255, 0, 0), 1)
                  cv2.imshow("Draw Region", clone)

          cv2.namedWindow("Draw Region")
          cv2.setMouseCallback("Draw Region", draw_polygon)

          while True:
              cv2.imshow("Draw Region", clone)
              key = cv2.waitKey(1) & 0xFF
              if key == 13:  # Enter
                  break
              elif key == 27:  # ESC
                  cv2.destroyAllWindows()
                  return

          cv2.destroyAllWindows()

          if len(points) > 2:
              mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
              cv2.fillPoly(mask, [np.array(points)], 255)
              white_bg = np.full_like(self.image, 255)
              result = np.where(mask[:, :, None] == 255, self.image, white_bg)
              x, y, w, h = cv2.boundingRect(np.array(points))
              self.cropped_image = result[y:y+h, x:x+w]
              self.show_image(self.cropped_image, "Cropped Image")
              self.save_result(self.cropped_image)
          else:
              print("Not enough points selected.")
     ```
     Here you convert the RGB image to Lab color space and extract only the L channel (lightness), which is useful for detecting e.g.folds in tissue. You are provided with a slider to interactively set a brightness threshold. Pixels below the threshold (i.e. folds) are removed and replaced with white. On window close, the current threshold is applied and the result is saved.
     ```python
     #python
      def extract_l_channel(self, img=None):
          if img is None:
              img = self.image
          lab = color.rgb2lab(img)
          return lab[:, :, 0]

      def interactive_L_threshold(self):
          L = self.extract_l_channel()
          fig, ax = plt.subplots()
          plt.subplots_adjust(bottom=0.25)
          ax.imshow(L, cmap='gray')
          plt.title('L Channel Thresholding')

          axthresh = plt.axes([0.25, 0.1, 0.65, 0.03])
          sthresh = Slider(axthresh, 'L threshold', 0.0, 100.0, valinit=80)

          def update(val):
              threshold = sthresh.val
              mask = L > threshold
              result = np.where(mask[:, :, None], self.image, 255).astype(np.uint8)
              ax.clear()
              ax.imshow(result)
              ax.set_title('Filtered Image')
              fig.canvas.draw_idle()

          def on_close(event):
              threshold = sthresh.val
              mask = L < threshold
              final_result = np.where(mask[:, :, None], self.image, 255).astype(np.uint8)
              self.save_result(final_result)

          sthresh.on_changed(update)
          fig.canvas.mpl_connect('close_event', on_close)
          plt.show(block=True)
     ```
     `process_images` opens a file dialog to let you select one or more images. For each selected image it creates a WSIProcessor instance and applies the chosen mode: 
     - "crop" → interactive_crop()
     - "lchannel" → interactive_L_threshold() 
     ```python
     #python
     # === GUI App ===
     def process_images(mode):
         file_paths = filedialog.askopenfilenames(title="Select Image(s)", filetypes=[("Image files", "*.jpg *.png *.jpeg *.tif *.tiff")])
         if not file_paths:
             return

         for path in file_paths:
             try:
                 processor = WSIProcessor(path)
                 if mode == "crop":
                     processor.interactive_crop()
                 elif mode == "lchannel":
                     processor.interactive_L_threshold()
             except Exception as e:
                 messagebox.showerror("Error", str(e))
     ```
     `start_gui` creates a simple Tkinter window with two buttons, "Manual Crop" or "L-Channel Filter", each launching `process_images()` in the selected mode. 
     ```python
     #python
     def start_gui():
         root = Tk()
         root.title("WSI Artifact Removal Tool")
         root.geometry("400x200")

         Label(root, text="Select Processing Mode", font=("Helvetica", 14)).pack(pady=20)

         Button(root, text="Manual Crop", command=lambda: process_images("crop"), width=30, height=2).pack(pady=5)
         Button(root, text="Automatic Filter", command=lambda: process_images("lchannel"), width=30, height=2).pack(pady=5)

         root.mainloop()
     ```
     When the script is run, it immediately launches the GUI as follows.
     ```python
     #python
      if __name__ == "__main__":
          start_gui()
     ```

     In summary, the execution flow is as follows.
      - User selects a processing mode through a graphical user interface
      - User picks one or more images
      - The tool processes each image interactively
      - Images are saved

     Now, with freshly cleaned images, you're ready to move into the next stages of the workflow.