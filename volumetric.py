import cv2
import numpy as np
import os
import vtk
from vtk.util import numpy_support

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

# Create volume mapper
volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
volume_mapper.SetInputData(vtk_image)

# Create volume properties
volume_property = vtk.vtkVolumeProperty()
volume_property.ShadeOn()
volume_property.SetInterpolationTypeToLinear()

# Create composite function
composite_function = vtk.vtkPiecewiseFunction()
composite_function.AddPoint(0, 0.0)
composite_function.AddPoint(255, 1.0)
volume_property.SetScalarOpacity(composite_function)

# Create color function
color_function = vtk.vtkColorTransferFunction()
color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
color_function.AddRGBPoint(255, 1.0, 1.0, 1.0)
volume_property.SetColor(color_function)

# Create volume actor
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Create renderer
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
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

