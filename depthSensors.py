import cv2
import numpy as np
import os

# Function to load MRI slices from a directory
def load_mri_slices(directory):
    slices = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpg'):  # Adjust this if your images are in a different format
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                slices.append(img)
    return slices

# Load the MRI slices
mri_slices = load_mri_slices('./Moderate Impairment/')
if not mri_slices:
    print("Error: No MRI slices loaded")
    exit()

# Convert to numpy array and check the shape
mri_volume = np.array(mri_slices)
print(f'MRI Volume shape: {mri_volume.shape}')

# Normalize the slices to the range [0, 255]
mri_volume = np.array([cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX) for slice in mri_volume], dtype=np.uint8)

# Generate synthetic depth map where depth corresponds to the slice index
depth_slices = np.arange(mri_volume.shape[0], dtype=np.float32)

# Normalize depth slices to the range [0, 255]
depth_slices = (depth_slices - np.min(depth_slices)) / (np.max(depth_slices) - np.min(depth_slices)) * 255
depth_slices = np.uint8(depth_slices)

# Create a depth map for each slice
depth_maps = np.stack([np.full_like(mri_volume[0], depth) for depth in depth_slices], axis=0)

# Generate 3D points from MRI slices and depth maps
points_3d = []
colors = []

for z, (slice, depth_map) in enumerate(zip(mri_volume, depth_maps)):
    height, width = slice.shape
    for y in range(height):
        for x in range(width):
            if slice[y, x] > 0:  # Only consider non-zero pixel values
                points_3d.append([x, y, depth_map[y, x]])
                colors.append([slice[y, x], slice[y, x], slice[y, x]])

points_3d = np.array(points_3d, dtype=np.float32)
colors = np.array(colors, dtype=np.uint8)

def write_ply(filename, points, colors):
    points = points.reshape(-1, 3)
    vertices = np.hstack([points, colors])
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, fmt='%f %f %f %d %d %d')

write_ply('mri_3d_model.ply', points_3d, colors)
print("3D model saved as mri_3d_model.ply")
