import cv2
import numpy as np

def power_law(image, gamma):
    # Normalize the pixel values to the range [0, 1]
    image_normalized = image / 255.0
    
    # Apply the power-law transformation
    image_gamma_corrected = np.power(image_normalized, gamma)
    
    # Denormalize the pixel values back to the range [0, 255]
    image_gamma_corrected = np.uint8(image_gamma_corrected * 255)
    
    return image_gamma_corrected


def contrast_stretching(image):
    # Compute the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Apply contrast stretching
    stretched_image = ((image - min_val) / (max_val - min_val)) * 255.0
    
    # Convert the pixel values to uint8 data type
    stretched_image = np.uint8(stretched_image)
    
    return stretched_image

def shear_vertical(image, shear_factor):
    """
    Shear the input image vertically while keeping its original size.
    
    Parameters:
        image (numpy.ndarray): Input image.
        shear_factor (float): Vertical shear factor. Positive values shear the image downwards, 
                              negative values shear the image upwards.
    
    Returns:
        numpy.ndarray: Sheared image with the same size as the input image.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define the transformation matrix for vertical shear
    shear_matrix = np.array([[1, shear_factor, 0],
                             [0, 1, 0]], dtype=np.float32)
    
    # Apply the shear transformation using warpAffine function
    sheared_image = cv2.warpAffine(image, shear_matrix, (width, height))
    
    return sheared_image

def rotate_image(image, angle):
    """
    Rotate the input image by the specified angle.
    
    Parameters:
        image (numpy.ndarray): Input image.
        angle (float): Angle of rotation in degrees. Positive values rotate the image clockwise,
                       while negative values rotate the image counter-clockwise.
    
    Returns:
        numpy.ndarray: Rotated image.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    
    # Apply the rotation transformation using warpAffine function
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

print("hello")
print("world")