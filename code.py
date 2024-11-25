import cv2
import numpy as np
def enhance_underwater_image(image_path):
   # Load the image
   img = cv2.imread(image_path)
   if img is None:
       raise FileNotFoundError("Image not found at the specified path!")
   # Step 1: White Balance Adjustment
   def white_balance(img):
       result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
       l, a, b = cv2.split(result)
       l = cv2.equalizeHist(l)
       result = cv2.merge((l, a, b))
       return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
   img = white_balance(img)
   # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
   lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   cl = clahe.apply(l)
   lab = cv2.merge((cl, a, b))
   img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   # Step 3: Denoising
   img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
   return img
# Input and output paths
input_path = "/DIPP111.jpeg"  # Replace with the actual image path
output_path = "enhanced_image.jpg"
# Enhance and save the image
try:
   enhanced_img = enhance_underwater_image(input_path)
   cv2.imwrite(output_path, enhanced_img)
except Exception as e:
   print(f"Error: {e}")
