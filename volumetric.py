import cv2
import numpy as np
import os
import vtk
from vtk.util import numpy_support
import re

# Function to load MRI slices from a directory with correct numerical sorting
def load_mri_slices(directory):
    slices = []
    # Regular expression to extract the slice number
    slice_pattern = re.compile(r'axial_slice(\d+)\.png')
    
    # Sort files based on the extracted numerical part
    sorted_filenames = sorted(
        os.listdir(directory), 
        key=lambda x: int(slice_pattern.search(x).group(1))
    )
    
    for filename in sorted_filenames:
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                slices.append(img)
    return slices

# Load the MRI slices
mri_slices = load_mri_slices('./12/')
# mri_slices = load_mri_slices('./Axial_slices/')
if not mri_slices:
    print("Error: No MRI slices loaded")
    exit()

# Convert to numpy array and check the shape
mri_volume = np.array(mri_slices)
print(f'MRI Volume shape: {mri_volume.shape}')

# Normalize the slices to the range [0, 255]
mri_volume = np.array([cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX) for slice in mri_volume], dtype=np.uint8)

# Check the shape of the volume
print(f'Volume shape: {mri_volume.shape}')  # Should be (number_of_slices, height, width)

# Convert the numpy array to a VTK image data object
def numpy_to_vtk_image(numpy_array):
    depth, height, width = numpy_array.shape
    vtk_data = numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, depth)
    vtk_image.SetSpacing(1, 1, 1)
    vtk_image.GetPointData().SetScalars(vtk_data)
    return vtk_image

# Create VTK image from numpy volume
vtk_image = numpy_to_vtk_image(mri_volume)

# Extract the surface using vtkMarchingCubes
marching_cubes = vtk.vtkMarchingCubes()
marching_cubes.SetInputData(vtk_image)
marching_cubes.SetValue(0, 80)  # Set the threshold value for the surface

# Create a polydata mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(marching_cubes.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0, 0, 0)

# Create render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Start the visualization
render_window.Render()
render_window_interactor.Start()

# Save the surface mesh to an .obj file
obj_writer = vtk.vtkOBJWriter()
obj_writer.SetFileName('mri_surface.obj')
obj_writer.SetInputConnection(marching_cubes.GetOutputPort())
obj_writer.Write()
