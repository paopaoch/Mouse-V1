import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

def generate_grating_image(width, height, num_stripes, orientation_degrees, contrast):
    # Generate a 1D stripe pattern along the x-axis
    x = np.linspace(-width/2, width/2, width)
    
    # Calculate the stripe pattern based on contrast
    stripe_pattern = ((num_stripes * x / width) % 2 < 1).astype(float)
    
    # Adjust contrast to interpolate between black and white (contrast=1.0) and mid-gray (contrast=0.5)
    stripe_pattern = contrast * stripe_pattern + (1 - contrast) * 0.5
    
    # Extend the stripe pattern to the full image height
    stripe_pattern = np.tile(stripe_pattern, (height, 1))
    
    # Rotate the stripe pattern to the desired orientation
    rotated_pattern = rotate(stripe_pattern, orientation_degrees, resize=True, mode='constant', cval=1)
    
    # Invert the rotated pattern to make gaps white instead of black
    grating_image = 1.0 - rotated_pattern
    
    # Display the grating image
    plt.figure(figsize=(8, 6))
    plt.imshow(grating_image, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.show()

# Example usage:
width = 400  # Width of the image
height = 300  # Height of the image
num_stripes = 20  # Number of black and white stripes
orientation_degrees = 45  # Orientation angle in degrees
contrast = 0.5  # Contrast value (0.0 for full gray, 1.0 for black and white)

# Generate a grating image with specified orientation and contrast
generate_grating_image(width, height, num_stripes, orientation_degrees, contrast)
