import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from tkinter import messagebox
import json
from tkinter import filedialog
from PIL import Image, ImageTk

# filter
def power_law(image, gamma):
    # Normalize the pixel values to the range [0, 1]
    image_normalized = image / 255.0
    
    # Apply the power-law transformation
    image_gamma_corrected = np.power(image_normalized, gamma)
    
    # Denormalize the pixel values back to the range [0, 255]
    image_gamma_corrected = np.uint8(image_gamma_corrected * 255)
    
    return image_gamma_corrected


def contrast_stretching(image, lower_threshold, upper_threshold):
    # Chuyển đổi ảnh sang dạng grayscale nếu cần
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Tìm giá trị pixel tối thiểu và tối đa trong ảnh
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)

    # Thực hiện contrast stretching
    stretched_image = np.where(gray_image < lower_threshold, 0,
                               np.where(gray_image > upper_threshold, 255,
                                        ((gray_image - lower_threshold) / (upper_threshold - lower_threshold)) * 255))

    # Chuyển đổi ảnh về dạng uint8
    stretched_image = np.uint8(stretched_image)

    return stretched_image


def shear_vertical(image, shear_factor):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create an empty array for the sheared image
    sheared_image = np.zeros_like(image)
    
    # Iterate over each pixel in the original image
    for y in range(height):
        for x in range(width):
            # Calculate the new y-coordinate for the sheared pixel
            new_y = y + shear_factor * x
            
            # Ensure the new y-coordinate is within the image boundaries
            if 0 <= new_y < height:
                # Copy the pixel value from the original image to the sheared image
                sheared_image[int(new_y), x] = image[y, x]
    
    return sheared_image


def shear_horizontal(image, shear_factor):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the amount of horizontal shear for each row
    shear_offset = int(shear_factor * height)
    
    # Create an empty array for the sheared image
    sheared_image = np.zeros_like(image)
    
    # Iterate over each pixel in the original image
    for y in range(height):
        # Calculate the new x-coordinate for the sheared pixel
        new_x = y * shear_factor
        
        # Shift the x-coordinate based on the shear offset
        new_x += shear_offset
        
        # Ensure the new x-coordinate is within the image boundaries
        if 0 <= new_x < width:
            # Copy the pixel value from the original image to the sheared image
            sheared_image[y, int(new_x)] = image[y, int(new_x)]
    
    return sheared_image


def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Convert the angle from degrees to radians
    theta = np.radians(angle)
    
    # Calculate the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0]
    ])
    
    # Define the center of rotation
    center_x = width / 2
    center_y = height / 2
    
    # Apply the rotation transformation
    rotated_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            # Map the pixel to the original coordinate system
            original_x = x - center_x
            original_y = y - center_y
            
            # Apply the rotation transformation
            new_x, new_y, _ = np.dot(rotation_matrix, [original_x, original_y, 1])
            
            # Map the pixel to the rotated coordinate system and round to the nearest integer
            new_x = int(round(new_x + center_x))
            new_y = int(round(new_y + center_y))
            
            # Ensure the new coordinates are within the image boundaries
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_image[new_y, new_x] = image[y, x]
    
    return rotated_image


def gaussian_filter(image, sigma):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    gaussian_filter = np.exp(-((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol) ** 2) / (2 * sigma ** 2)) * (1-np.exp(-((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol) ** 2) / (2 * sigma ** 2)))
    
    f_transform_filtered_gaussian = f_shift * gaussian_filter
    image_filtered_gaussian = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered_gaussian)).real
    return image_filtered_gaussian


def butterworth_filter(image, D0, n):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    butterworth_filter = 1 / (1 + ((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol) ** 2) / D0 ** 2) ** n

    f_transform_filtered_butterworth = f_shift * butterworth_filter
    image_filtered_butterworth = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered_butterworth)).real
    
    return image_filtered_butterworth


def add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, amount):
    # amount -> % of the pixels affected by noise (5% -> amount = 0.005)
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper_ratio))

    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image


def grayscale_formula(image):
    # Extract the color channels
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    # Convert each pixel to grayscale using the luminance formula
    grayscale_image = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    
    # Convert to uint8 data type
    grayscale_image = np.uint8(grayscale_image)
    
    return grayscale_image


def median_blur(image, kernel_size):
    # Áp dụng median blur
    blurred_image = cv2.medianBlur(image, kernel_size)
    
    return blurred_image


# read image

def read_images_in_folder(folder_path):
    image_list = []
    
    # Lấy danh sách tất cả các tệp trong thư mục
    file_list = os.listdir(folder_path)
    
    # Lặp qua từng tệp và đọc ảnh
    for file_name in file_list:
        # Kiểm tra nếu tệp là ảnh (có thể cần xử lý kiểu định dạng ảnh phù hợp)
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            if image is not None:
                image_list.append(image)
            else:
                print(f"Không thể đọc ảnh từ {image_path}")
    
    return image_list


def save_to_new_folder(image, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, filename)
    cv2.imwrite(output_path, image)


def process_data(image_path, folder_path, gamma, lower_threshold, upper_threshold, shear_factor, angle, sigma, D0, n, ratio, amount, kernel_size):
    image_list = read_images_in_folder(image_path)
    for image in image_list:
        # convert image
        img_gamma = power_law(image, gamma)
        img_contrast = contrast_stretching(image, lower_threshold, upper_threshold)
        img_shear_vertical = shear_vertical(image, shear_factor)
        img_shear_horizontal = shear_horizontal(image, shear_factor)
        img_rotate = rotate_image(image, angle)
        img_gaussian = gaussian_filter(image, sigma)
        img_butterworth = butterworth_filter(image, D0, n)
        img_salt_and_pepper = add_salt_and_pepper_noise(image, ratio, amount)
        img_grayscale = grayscale_formula(image)
        img_medianblur = median_blur(image, kernel_size)

        # process new data
        save_to_new_folder(img_gamma, folder_path)
        save_to_new_folder(img_shear_vertical, folder_path)
        save_to_new_folder(img_contrast, folder_path)
        save_to_new_folder(img_shear_horizontal, folder_path)
        save_to_new_folder(img_rotate, folder_path)
        save_to_new_folder(img_gaussian, folder_path)
        save_to_new_folder(img_butterworth, folder_path)
        save_to_new_folder(img_salt_and_pepper, folder_path)
        save_to_new_folder(img_grayscale, folder_path)
        save_to_new_folder(img_medianblur, folder_path)

