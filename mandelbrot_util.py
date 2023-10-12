import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mandelbrot_gpu = None
block_size = (16, 16, 1)
grid_size = None
output_gpu = None

def init():
    global mandelbrot_gpu
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

    mod = SourceModule(mandelbrot_kernel)
    mandelbrot_gpu = mod.get_function("mandelbrot")


def make_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iterations):
    global output_gpu, block_size, grid_size, mandelbrot_gpu
    output_gpu = cuda.mem_alloc(width * height * 4)
    grid_size = (width // block_size[0] + 1, height // block_size[1] + 1, 1)

    mandelbrot_gpu(output_gpu, np.int32(width), np.int32(height), np.double(x_min), np.double(x_max),
                   np.double(y_min), np.double(y_max), np.int32(max_iterations),
                   block=block_size, grid=grid_size)

    output_cpu = np.empty((height, width), dtype=np.int32)
    cuda.memcpy_dtoh(output_cpu, output_gpu)
    output_normalized = (255 * (output_cpu - np.min(output_cpu)) / np.ptp(output_cpu)).astype(np.uint8)

    output_gpu.free()

    return output_normalized