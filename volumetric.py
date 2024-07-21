import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load DICOM images using SimpleITK
def load_dicom_images(folder_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)

# Segment images (simple thresholding for illustration)
def segment_images(images):
    segmented = images > 100  # Adjust threshold as needed
    return segmented

# Visualize 3D model
def visualize_3d(segmented):
    verts, faces, _, _ = measure.marching_cubes(segmented, level=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)
    ax.set_xlim(0, segmented.shape[0])
    ax.set_ylim(0, segmented.shape[1])
    ax.set_zlim(0, segmented.shape[2])
    plt.show()

# Main function
folder_path = 'path_to_dicom_folder'
images = load_dicom_images(folder_path)
segmented = segment_images(images)
visualize_3d(segmented)
