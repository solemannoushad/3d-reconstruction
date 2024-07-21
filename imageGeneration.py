import cv2
import numpy as np

def generate_synthetic_mri_images(width=256, height=256):
    # Create a synthetic base image (simple gradient for demonstration)
    base_image = np.zeros((height, width), np.uint8)
    cv2.rectangle(base_image, (50, 50), (width-50, height-50), 255, -1)
    cv2.circle(base_image, (width//2, height//2), 40, 127, -1)

    # Create left and right images by applying horizontal shift
    shift_amount = 5  # Pixel shift for stereo effect
    left_image = np.roll(base_image, -shift_amount, axis=1)
    right_image = np.roll(base_image, shift_amount, axis=1)

    return left_image, right_image

left_image, right_image = generate_synthetic_mri_images()

# Save images for visualization
cv2.imwrite('left_mri_image.png', left_image)
cv2.imwrite('right_mri_image.png', right_image)

# Display images
cv2.imshow('Left Image', left_image)
cv2.imshow('Right Image', right_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
