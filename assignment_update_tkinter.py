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

root = tk.Tk()
root.geometry('1080x720')
root.title('process data')

image_path = r"1.jpg"
img_rgb = cv2.imread(image_path)
gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def contrast_page():
    global lower_scale, upper_scale, lower_entry, upper_entry, img_label, gray_image

    contrast_frame = tk.Frame(main_frame)

    def contrast_stretching(image, lower_threshold, upper_threshold):

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

    def update_image(lower_threshold, upper_threshold, img_label):
        global gray_image
        stretched_image = contrast_stretching(gray_image, lower_threshold, upper_threshold)
        img = Image.fromarray(stretched_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(value):
        lower = lower_scale.get()
        upper = upper_scale.get()
        lower_entry.delete(0, tk.END)
        upper_entry.delete(0, tk.END)
        lower_entry.insert(0, str(lower))
        upper_entry.insert(0, str(upper))
        update_image(lower, upper, img_label)

    def on_entry_change(value):
        lower = int(lower_entry.get())
        upper = int(upper_entry.get())
        lower_scale.set(lower)
        upper_scale.set(upper)
        update_image(lower, upper, img_label)


    # Tạo thanh trượt và ô nhập số cho contrast stretching
    lower_scale = ttk.Scale(contrast_frame, from_=0, to=255, orient="horizontal", length=200, command=on_scale_change)
    lower_scale.set(0)
    lower_scale.pack(padx=10, pady=5)

    upper_scale = ttk.Scale(contrast_frame, from_=0, to=255, orient="horizontal", length=200, command=on_scale_change)
    upper_scale.set(255)
    upper_scale.pack(padx=10, pady=5)

    lower_frame = ttk.Frame(contrast_frame)
    lower_frame.pack(padx=10, pady=5)
    lower_label = ttk.Label(lower_frame, text="Lower Threshold:")
    lower_label.grid(row=0, column=0)
    lower_entry = ttk.Entry(lower_frame)
    lower_entry.grid(row=0, column=1)
    lower_entry.insert(0, str(lower_scale.get()))
    lower_entry.bind("<Return>", on_entry_change)

    upper_frame = ttk.Frame(contrast_frame)
    upper_frame.pack(padx=10, pady=5)
    upper_label = ttk.Label(upper_frame, text="Upper Threshold:")
    upper_label.grid(row=0, column=0)
    upper_entry = ttk.Entry(upper_frame)
    upper_entry.grid(row=0, column=1)
    upper_entry.insert(0, str(upper_scale.get()))
    upper_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(contrast_frame)
    img_label.pack(padx=10, pady=5)

    update_image(lower_scale.get(), upper_scale.get(), img_label)
    contrast_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung contrast vào main_frame
# ____________________________________________________________________________________
def power_law_page():
    global gamma_scale, gamma_entry, img_label, img_rgb, gray_image
    
    power_law_frame = tk.Frame(main_frame)

    def power_law(image, gamma):
        # Normalize the pixel values to the range [0, 1]
        image_normalized = image / 255.0
        
        # Apply the power-law transformation
        image_gamma_corrected = np.power(image_normalized, gamma)
        
        # Denormalize the pixel values back to the range [0, 255]
        image_gamma_corrected = np.uint8(image_gamma_corrected * 255)
        
        return image_gamma_corrected

    def update_image(gamma):
        global img_label, gray_image
        gamma_corrected_image = power_law(gray_image, gamma)
        img = Image.fromarray(gamma_corrected_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        gamma = gamma_scale.get()
        gamma_entry.delete(0, tk.END)
        gamma_entry.insert(0, str(gamma))
        update_image(gamma)

    def on_entry_change(event):
        gamma = float(gamma_entry.get())
        gamma_scale.set(gamma)
        update_image(gamma)

    
    # Load ảnh gốc

    # Tạo thanh trượt cho chỉ số gamma
    gamma_scale = ttk.Scale(power_law_frame, from_=0.1, to=5.0, orient="horizontal", length=200, command=on_scale_change)
    gamma_scale.set(1.0)
    gamma_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho chỉ số gamma
    gamma_frame = ttk.Frame(power_law_frame)
    gamma_frame.pack(padx=10, pady=5)
    gamma_label = ttk.Label(gamma_frame, text="Gamma:")
    gamma_label.grid(row=0, column=0)
    gamma_entry = ttk.Entry(gamma_frame)
    gamma_entry.grid(row=0, column=1)
    gamma_entry.insert(0, str(gamma_scale.get()))
    gamma_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(power_law_frame)
    img_label.pack(padx=10, pady=5)

    update_image(gamma_scale.get())
    power_law_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung power_law vào main_frame
# ____________________________________________________________________________________
def shear_vertical_page():
    global shear_scale, shear_entry, img_label, gray_image
    
    shear_vertical_frame = tk.Frame(main_frame)

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

    def update_image(shear_factor):
        global img_label, gray_image
        sheared_image = shear_vertical(gray_image, shear_factor)
        img = Image.fromarray(sheared_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        shear_factor = shear_scale.get()
        shear_entry.delete(0, tk.END)
        shear_entry.insert(0, str(shear_factor))
        update_image(shear_factor)

    def on_entry_change(event):
        shear_factor = float(shear_entry.get())
        shear_scale.set(shear_factor)
        update_image(shear_factor)

    # Tạo thanh trượt cho hệ số cắt dọc
    shear_scale = ttk.Scale(shear_vertical_frame, from_=-1.0, to=1.0, orient="horizontal", length=200, command=on_scale_change)
    shear_scale.set(0.0)
    shear_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho hệ số cắt dọc
    shear_frame = ttk.Frame(shear_vertical_frame)
    shear_frame.pack(padx=10, pady=5)
    shear_label = ttk.Label(shear_frame, text="Shear Factor:")
    shear_label.grid(row=0, column=0)
    shear_entry = ttk.Entry(shear_frame)
    shear_entry.grid(row=0, column=1)
    shear_entry.insert(0, str(shear_scale.get()))
    shear_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(shear_vertical_frame)
    img_label.pack(padx=10, pady=5)

    update_image(shear_scale.get())
    shear_vertical_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung shear_vertical vào main_frame
# ____________________________________________________________________________________
def shear_horizontal_page():
    global shear_scale, shear_entry, img_label, gray_image
    
    shear_horizontal_frame = tk.Frame(main_frame)
    
    def shear_horizontal(image, shear_factor):
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate the amount of horizontal shear for each row
        shear_offset = int(shear_factor * height)
        
        # Create an empty array for the sheared image
        sheared_image = np.zeros_like(image)
        
        # Iterate over each pixel in the sheared image
        for y in range(height):
            for x in range(width):
                # Calculate the new x-coordinate for the sheared pixel
                new_x = x + shear_factor * (y - shear_offset)
                
                # Ensure the new x-coordinate is within the image boundaries
                if 0 <= new_x < width:
                    # Copy the pixel value from the original image to the sheared image
                    sheared_image[y, int(new_x)] = image[y, x]
        
        return sheared_image


    def update_image(shear_factor):
        global img_label, gray_image
        sheared_image = shear_horizontal(gray_image, shear_factor)
        img = Image.fromarray(sheared_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        shear_factor = shear_scale.get()
        shear_entry.delete(0, tk.END)
        shear_entry.insert(0, str(shear_factor))
        update_image(shear_factor)

    def on_entry_change(event):
        shear_factor = float(shear_entry.get())
        shear_scale.set(shear_factor)
        update_image(shear_factor)

    # Tạo thanh trượt cho hệ số cắt ngang
    shear_scale = ttk.Scale(shear_horizontal_frame, from_=-1.0, to=1.0, orient="horizontal", length=200, command=on_scale_change)
    shear_scale.set(0.0)
    shear_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho hệ số cắt ngang
    shear_frame = ttk.Frame(shear_horizontal_frame)
    shear_frame.pack(padx=10, pady=5)
    shear_label = ttk.Label(shear_frame, text="Shear Factor:")
    shear_label.grid(row=0, column=0)
    shear_entry = ttk.Entry(shear_frame)
    shear_entry.grid(row=0, column=1)
    shear_entry.insert(0, str(shear_scale.get()))
    shear_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(shear_horizontal_frame)
    img_label.pack(padx=10, pady=5)

    update_image(shear_scale.get())
    shear_horizontal_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung shear_horizontal vào main_frame
# ____________________________________________________________________________________
def rotate_image_page():
    global angle_scale, angle_entry, img_label, gray_image
    
    rotate_frame = tk.Frame(main_frame)

    def rotate_image(image, angle):
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert the angle from degrees to radians
        theta = np.radians(angle)
        
        # Calculate the rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]  # Thêm giá trị z (hoặc z-index)
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

    def update_image(angle):
        global img_label, gray_image
        rotated_image = rotate_image(gray_image, angle)
        img = Image.fromarray(rotated_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        angle = angle_scale.get()
        angle_entry.delete(0, tk.END)
        angle_entry.insert(0, str(angle))
        update_image(angle)

    def on_entry_change(event):
        angle = float(angle_entry.get())
        angle_scale.set(angle)
        update_image(angle)

    # Tạo thanh trượt cho góc quay
    angle_scale = ttk.Scale(rotate_frame, from_=-180, to=180, orient="horizontal", length=200, command=on_scale_change)
    angle_scale.set(0)
    angle_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho góc quay
    angle_frame = ttk.Frame(rotate_frame)
    angle_frame.pack(padx=10, pady=5)
    angle_label = ttk.Label(angle_frame, text="Angle (degrees):")
    angle_label.grid(row=0, column=0)
    angle_entry = ttk.Entry(angle_frame)
    angle_entry.grid(row=0, column=1)
    angle_entry.insert(0, str(angle_scale.get()))
    angle_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(rotate_frame)
    img_label.pack(padx=10, pady=5)

    update_image(angle_scale.get())
    rotate_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung rotate vào main_frame
# ____________________________________________________________________________________
def gaussian_filter_page():
    global sigma_scale, sigma_entry, img_label, gray_image
    
    filter_frame = tk.Frame(main_frame)
    def gaussian_filter(image, sigma):
        # Convert image to frequency domain
        f_img = np.fft.fft2(image)
        fshift = np.fft.fftshift(f_img)

        # Calculate dimensions of frequency domain image
        rows, cols = f_img.shape

        # Calculate center of frequency domain image
        crow, ccol = rows // 2, cols // 2

        # Generate Gaussian filter mask based on frequency domain dimensions
        gaussian_filter = np.exp(-((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol) ** 2) / (2 * sigma ** 2))

        # Apply Gaussian filter in frequency domain
        filtered_fshift = fshift * gaussian_filter

        # Convert filtered image back to spatial domain
        filtered_f_img = np.fft.ifftshift(filtered_fshift)
        filtered_img = np.fft.ifft2(filtered_f_img)
        filtered_img = np.abs(filtered_img)

        return filtered_img

    def update_image(sigma):
        global img_label, gray_image
        filtered_image = gaussian_filter(gray_image, sigma)
        img = Image.fromarray(filtered_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        sigma = sigma_scale.get()
        sigma_entry.delete(0, tk.END)
        sigma_entry.insert(0, str(sigma))
        update_image(sigma)

    def on_entry_change(event):
        sigma = float(sigma_entry.get())
        sigma_scale.set(sigma)
        update_image(sigma)

    # Tạo thanh trượt cho sigma
    sigma_scale = ttk.Scale(filter_frame, from_=20, to=40, orient="horizontal", length=200, command=on_scale_change)
    sigma_scale.set(1)
    sigma_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho sigma
    sigma_frame = ttk.Frame(filter_frame)
    sigma_frame.pack(padx=10, pady=5)
    sigma_label = ttk.Label(sigma_frame, text="Sigma:")
    sigma_label.grid(row=0, column=0)
    sigma_entry = ttk.Entry(sigma_frame)
    sigma_entry.grid(row=0, column=1)
    sigma_entry.insert(0, str(sigma_scale.get()))
    sigma_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(filter_frame)
    img_label.pack(padx=10, pady=5)

    update_image(sigma_scale.get())
    filter_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung filter vào main_frame
# ____________________________________________________________________________________
def butterworth_filter_page():
    global D0_scale, D0_entry, n_scale, n_entry, img_label, gray_image
    
    filter_frame = tk.Frame(main_frame)
    def butterworth_filter(image, D0, n):
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        height, width = image.shape[:2]  # Sử dụng height và width từ ảnh

        crow, ccol = height // 2, width // 2

        butterworth_filter = 1 / (1 + ((np.arange(height) - crow) ** 2 + (np.arange(width) - ccol) ** 2) / D0 ** 2) ** n

        f_transform_filtered_butterworth = f_shift * butterworth_filter
        image_filtered_butterworth = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered_butterworth)).real
        
        return image_filtered_butterworth

    def update_image(D0, n):
        global img_label, gray_image
        filtered_image = butterworth_filter(gray_image, D0, n)
        img = Image.fromarray(filtered_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        D0 = D0_scale.get()
        n = n_scale.get()
        D0_entry.delete(0, tk.END)
        D0_entry.insert(0, str(D0))
        n_entry.delete(0, tk.END)
        n_entry.insert(0, str(n))
        update_image(D0, n)

    def on_entry_change(event):
        D0 = float(D0_entry.get())
        n = float(n_entry.get())
        D0_scale.set(D0)
        n_scale.set(n)
        update_image(D0, n)

    # Tạo thanh trượt cho D0 và n
    D0_scale = ttk.Scale(filter_frame, from_=1, to=100, orient="horizontal", length=200, command=on_scale_change)
    D0_scale.set(10)
    D0_scale.pack(padx=10, pady=5)

    n_scale = ttk.Scale(filter_frame, from_=1, to=10, orient="horizontal", length=200, command=on_scale_change)
    n_scale.set(2)
    n_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho D0 và n
    D0_frame = ttk.Frame(filter_frame)
    D0_frame.pack(padx=10, pady=5)
    D0_label = ttk.Label(D0_frame, text="D0:")
    D0_label.grid(row=0, column=0)
    D0_entry = ttk.Entry(D0_frame)
    D0_entry.grid(row=0, column=1)
    D0_entry.insert(0, str(D0_scale.get()))
    D0_entry.bind("<Return>", on_entry_change)

    n_frame = ttk.Frame(filter_frame)
    n_frame.pack(padx=10, pady=5)
    n_label = ttk.Label(n_frame, text="n:")
    n_label.grid(row=0, column=0)
    n_entry = ttk.Entry(n_frame)
    n_entry.grid(row=0, column=1)
    n_entry.insert(0, str(n_scale.get()))
    n_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(filter_frame)
    img_label.pack(padx=10, pady=5)

    update_image(D0_scale.get(), n_scale.get())
    filter_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung filter vào main_frame
# ____________________________________________________________________________________
def salt_and_pepper_noise_page():
    global salt_vs_pepper_ratio_scale, amount_scale, salt_vs_pepper_ratio_entry, amount_entry, img_label, img_rgb
    
    noise_frame = tk.Frame(main_frame)
    
    def add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, amount):
        noisy_image = np.copy(image)
        num_pixels = image.size
        
        # Tính số lượng pixel nhiễu muối và tiêu
        num_salt = int(np.ceil(amount * num_pixels * salt_vs_pepper_ratio))
        num_pepper = int(np.ceil(amount * num_pixels * (1.0 - salt_vs_pepper_ratio)))

        # Thêm nhiễu muối
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 255

        # Thêm nhiễu tiêu
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 0

        return noisy_image

    def update_image(salt_vs_pepper_ratio, amount):
        global img_label, gray_image
        noisy_image = add_salt_and_pepper_noise(gray_image, salt_vs_pepper_ratio, amount)
        img = Image.fromarray(noisy_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        salt_vs_pepper_ratio = salt_vs_pepper_ratio_scale.get()
        amount = amount_scale.get()
        salt_vs_pepper_ratio_entry.delete(0, tk.END)
        salt_vs_pepper_ratio_entry.insert(0, str(salt_vs_pepper_ratio))
        amount_entry.delete(0, tk.END)
        amount_entry.insert(0, str(amount))
        update_image(salt_vs_pepper_ratio, amount)

    def on_entry_change(event):
        salt_vs_pepper_ratio = float(salt_vs_pepper_ratio_entry.get())
        amount = float(amount_entry.get())
        salt_vs_pepper_ratio_scale.set(salt_vs_pepper_ratio)
        amount_scale.set(amount)
        update_image(salt_vs_pepper_ratio, amount)


    # Tạo thanh trượt cho tỷ lệ muối và tiêu
    salt_vs_pepper_ratio_scale = ttk.Scale(noise_frame, from_=0, to=1, orient="horizontal", length=200, command=on_scale_change)
    salt_vs_pepper_ratio_scale.set(0.5)
    salt_vs_pepper_ratio_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho tỷ lệ muối và tiêu
    salt_vs_pepper_ratio_frame = ttk.Frame(noise_frame)
    salt_vs_pepper_ratio_frame.pack(padx=10, pady=5)
    salt_vs_pepper_ratio_label = ttk.Label(salt_vs_pepper_ratio_frame, text="Salt vs Pepper Ratio:")
    salt_vs_pepper_ratio_label.grid(row=0, column=0)
    salt_vs_pepper_ratio_entry = ttk.Entry(salt_vs_pepper_ratio_frame)
    salt_vs_pepper_ratio_entry.grid(row=0, column=1)
    salt_vs_pepper_ratio_entry.insert(0, str(salt_vs_pepper_ratio_scale.get()))
    salt_vs_pepper_ratio_entry.bind("<Return>", on_entry_change)

    # Tạo thanh trượt cho lượng nhiễu
    amount_scale = ttk.Scale(noise_frame, from_=0, to=0.1, orient="horizontal", length=200, command=on_scale_change)
    amount_scale.set(0.05)
    amount_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho lượng nhiễu
    amount_frame = ttk.Frame(noise_frame)
    amount_frame.pack(padx=10, pady=5)
    amount_label = ttk.Label(amount_frame, text="Noise Amount:")
    amount_label.grid(row=0, column=0)
    amount_entry = ttk.Entry(amount_frame)
    amount_entry.grid(row=0, column=1)
    amount_entry.insert(0, str(amount_scale.get()))
    amount_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(noise_frame)
    img_label.pack(padx=10, pady=5)

    update_image(salt_vs_pepper_ratio_scale.get(), amount_scale.get())
    noise_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung noise vào main_frame
# ____________________________________________________________________________________
def grayscale_formula_page():
    global img_label, img_rgb
    
    grayscale_frame = tk.Frame(main_frame)
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

    def update_image_original():
        global img_label, img_rgb
        img = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def update_image_grayscale():
        global img_label, img_rgb
        grayscale_image = grayscale_formula(img_rgb)
        img = Image.fromarray(grayscale_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    # Hiển thị ảnh gốc
    original_button = tk.Button(grayscale_frame, text="Original Image", command=update_image_original)
    original_button.pack(padx=10, pady=5)

    # Hiển thị ảnh xám
    grayscale_button = tk.Button(grayscale_frame, text="Grayscale Image", command=update_image_grayscale)
    grayscale_button.pack(padx=10, pady=5)

    # Hiển thị ảnh gốc ban đầu
    img_label = tk.Label(grayscale_frame)
    img_label.pack(padx=10, pady=5)

    update_image_original()
    grayscale_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung Grayscale vào main_frame
# ____________________________________________________________________________________
def median_blur_page():
    global kernel_scale, kernel_entry, img_label, gray_image
    
    blur_frame = tk.Frame(main_frame)

    def median_blur(image, kernel_size):
        kernel_size = int(kernel_size)
        # Áp dụng Median Blur
        blurred_image = cv2.medianBlur(image, kernel_size)
        return blurred_image

    def update_image(kernel_size):
        global img_label, gray_image
        blurred_image = median_blur(gray_image, kernel_size)
        img = Image.fromarray(blurred_image)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def on_scale_change(event):
        kernel_size = kernel_scale.get()
        kernel_entry.delete(0, tk.END)
        kernel_entry.insert(0, str(kernel_size))
        update_image(kernel_size)

    def on_entry_change(event):
        kernel_size = int(kernel_entry.get())
        kernel_scale.set(kernel_size)
        update_image(kernel_size)

    # Load ảnh gốc
    image_path = r"C:\Users\ndhdu\Downloads\43eb02fd44a7e8f9b1b6.jpg"
    img_rgb = cv2.imread(image_path)

    # Tạo thanh trượt cho kích thước kernel
    kernel_scale = ttk.Scale(blur_frame, from_=3, to=15, orient="horizontal", length=200, command=on_scale_change)
    kernel_scale.set(3)
    kernel_scale.pack(padx=10, pady=5)

    # Tạo ô nhập số cho kích thước kernel
    kernel_frame = ttk.Frame(blur_frame)
    kernel_frame.pack(padx=10, pady=5)
    kernel_label = ttk.Label(kernel_frame, text="Kernel Size:")
    kernel_label.grid(row=0, column=0)
    kernel_entry = ttk.Entry(kernel_frame)
    kernel_entry.grid(row=0, column=1)
    kernel_entry.insert(0, str(kernel_scale.get()))
    kernel_entry.bind("<Return>", on_entry_change)

    # Hiển thị ảnh gốc
    img_label = tk.Label(blur_frame)
    img_label.pack(padx=10, pady=5)

    update_image(kernel_scale.get())
    blur_frame.pack(pady=10, padx=10, side=tk.LEFT)  # Đặt khung blur vào main_frame

def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()

def indicate(page):
    delete_pages()
    page()

option_frame = tk.Frame(root, bg='#c3c3c3')

contrast_button = tk.Button(option_frame, text='Contrast', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(contrast_page))
contrast_button.place(x=20, y=50)

contrast_button = tk.Button(option_frame, text='Power Law', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(power_law_page))
contrast_button.place(x=20, y=100)

contrast_button = tk.Button(option_frame, text='Shear Vertical', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(shear_vertical_page))
contrast_button.place(x=20, y=150)

contrast_button = tk.Button(option_frame, text='Shear Horizontal', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(shear_horizontal_page))
contrast_button.place(x=20, y=200)

contrast_button = tk.Button(option_frame, text='Rotation', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(rotate_image_page))
contrast_button.place(x=20, y=250)

contrast_button = tk.Button(option_frame, text='Gaussian(freq domain)', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(gaussian_filter_page))
contrast_button.place(x=20, y=300)

contrast_button = tk.Button(option_frame, text='Butterworth(freq domain)', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(butterworth_filter_page))
contrast_button.place(x=20, y=350)

contrast_button = tk.Button(option_frame, text='Salt and Pepper(freq domain)', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(salt_and_pepper_noise_page))
contrast_button.place(x=20, y=400)

contrast_button = tk.Button(option_frame, text='Grayscale', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(grayscale_formula_page))
contrast_button.place(x=20, y=450)

contrast_button = tk.Button(option_frame, text='Median Blur', font=('Bold', 15), fg='#158aff', bd=0, bg='#c3c3c3', command=lambda: indicate(median_blur_page))
contrast_button.place(x=20, y=500)

option_frame.pack(side=tk.LEFT)
option_frame.pack_propagate(False)
option_frame.configure(width=350, height=720)

main_frame = tk.Frame(root)
main_frame.pack(side=tk.LEFT)
main_frame.pack_propagate(False)
main_frame.configure(width=1080, height=720)

root.mainloop()
