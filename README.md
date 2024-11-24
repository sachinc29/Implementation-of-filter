# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import necessary libraries (cv2, NumPy, Matplotlib) for image loading, filtering, and visualization.

### Step 2: 
Load the image using cv2.imread() and convert it to RGB format using cv2.cvtColor() for proper display in Matplotlib.

### Step 3: 
Apply different filters:
1. Averaging Filter: Define an averaging kernel using np.ones() and apply it to the image using cv2.filter2D().
2. Weighted Averaging Filter: Define a weighted kernel (e.g., 3x3 Gaussian-like) and apply it with cv2.filter2D().
3. Gaussian Filter: Use cv2.GaussianBlur() to apply Gaussian blur.
4. Median Filter: Use cv2.medianBlur() to reduce noise.
5. Laplacian Operator: Use cv2.Laplacian() to apply edge detection.
    

### Step 4: 
Display each filtered image using plt.subplot() and plt.imshow() for side-by-side comparison of the original and processed images.

### Step 5: 
Save or show the images using plt.show() after applying each filter to visualize the effects of smoothing and sharpening.


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("spidy.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(7, 3))
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
```
![image](https://github.com/user-attachments/assets/eb9ce6a9-c246-42f5-b854-bfc7146e4e14)

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(7, 3))
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/783dd248-b8d5-462e-b84b-cb8c908ad133)

ii) Using Weighted Averaging Filter
```Python
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(7, 3))
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/d57b3164-a5b1-4a05-ae5e-b19a928940e6)

iii) Using Gaussian Filter
```Python
gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)

plt.figure(figsize=(7, 3))
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/8bb9a9bc-6449-4616-aaf9-667bf5f1aa93)

iv)Using Median Filter
```Python
median_blur = cv2.medianBlur(image2, 11)

plt.figure(figsize=(7, 3))
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()

```
![image](https://github.com/user-attachments/assets/645a8df9-e0cf-41d5-b889-c7345ac525b7)

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
sharpened_image = cv2.filter2D(image2, -1, kernel1)

plt.figure(figsize=(7, 3))
plt.imshow(sharpened_image)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()
```
![image](https://github.com/user-attachments/assets/e7052f24-f515-439f-99a3-c1f58925904f)

ii) Using Laplacian Operator
```Python
laplacian = cv2.Laplacian(image2, cv2.CV_64F)

plt.figure(figsize=(7, 3))
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()

```
![image](https://github.com/user-attachments/assets/0cd3e59e-f80a-4de0-b1d4-3e8c0b55e111)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
