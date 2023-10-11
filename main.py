import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image

# Define the parameters for the Mandelbrot set
width, height = 24000, 24000
# width, height = 1920, 1080
max_iterations = 2047
resolution = 0.00000001
center_x = -1.484585267
center_y = 0

aspect = (height / width)

x_min = center_x - resolution
x_max = center_x + resolution
y_min = center_y - (resolution * aspect)
y_max = center_y + (resolution * aspect)


# Create a grid of complex numbers
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
c = X + 1j * Y

# Create a CUDA kernel to compute the Mandelbrot set
mandelbrot_kernel = """
__global__ void mandelbrot(int *output, int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        double x = x_min + idx * (x_max - x_min) / (double)(width - 1);
        double y = y_min + idy * (y_max - y_min) / (double)(height - 1);
        double2 z = make_double2(x, y);
        double2 c = z;
        int iteration = 0;
        while (iteration < max_iterations && z.x * z.x + z.y * z.y < 4.0) {
            double temp = z.x * z.x - z.y * z.y + c.x;
            z.y = 2.0 * z.x * z.y + c.y;
            z.x = temp;
            iteration++;
        }
        output[idy * width + idx] = iteration;
    }
}
"""

# Compile the CUDA kernel
mod = SourceModule(mandelbrot_kernel)

# Get a reference to the CUDA kernel function
mandelbrot_gpu = mod.get_function("mandelbrot")

# Create a GPU array to store the Mandelbrot set data
output_gpu = cuda.mem_alloc(width * height * 4)

# Define thread block and grid dimensions
block_size = (16, 16, 1)  # Specify a third dimension for the block size
grid_size = (width // block_size[0] + 1, height // block_size[1] + 1, 1)

# Compute the Mandelbrot set on the GPU
mandelbrot_gpu(output_gpu, np.int32(width), np.int32(height), np.double(x_min), np.double(x_max),
               np.double(y_min), np.double(y_max), np.int32(max_iterations),
               block=block_size, grid=grid_size)

# Copy the result back to the CPU
output_cpu = np.empty((height, width), dtype=np.int32)
cuda.memcpy_dtoh(output_cpu, output_gpu)

# Normalize the Mandelbrot set data to the range [0, 255]
output_normalized = (255 * (output_cpu - np.min(output_cpu)) / np.ptp(output_cpu)).astype(np.uint8)

# Convert the Mandelbrot set data to an image
mandelbrot_image = Image.fromarray(output_normalized)

# Save the image to a file
mandelbrot_image.save("mandelbrot_set.png")

# Display the image
mandelbrot_image.show()

output_gpu.free()
