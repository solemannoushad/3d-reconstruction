import cv2
import numpy as np

# Load the synthetic images generated
left_image = cv2.imread('left_mri_image.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right_mri_image.png', cv2.IMREAD_GRAYSCALE)

# Preprocessing
left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
right_image = cv2.GaussianBlur(right_image, (5, 5), 0)

# Stereo Matching
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_image, right_image)
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)
cv2.imshow('Disparity', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Depth Map Generation
focal_length = 700
baseline = 0.1
disparity[disparity == 0] = 0.1
depth_map = (focal_length * baseline) / disparity
depth_map = cv2.normalize(depth_map, depth_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.float32)
depth_map = np.uint8(depth_map)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3D Reconstruction
camera_matrix = np.array([[focal_length, 0, left_image.shape[1] / 2],
                          [0, focal_length, left_image.shape[0] / 2],
                          [0, 0, 1]])
height, width = depth_map.shape
Q = np.float32([[1, 0, 0, -width / 2],
                [0, -1, 0, height / 2],
                [0, 0, 0, -focal_length],
                [0, 0, 1 / baseline, 0]])

points_3d = cv2.reprojectImageTo3D(disparity, Q)
mask_map = disparity > disparity.min()
output_points = points_3d[mask_map]
output_colors = left_image[mask_map]

# Ensure the colors have the shape (Nx3)
output_colors = np.stack((output_colors,) * 3, axis=-1).reshape(-1, 3)

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

write_ply('output.ply', output_points, output_colors)
print("3D model saved as output.ply")
