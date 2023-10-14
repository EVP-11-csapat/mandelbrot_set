import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit


class Mandelbrot:
    def __init__(self, width: int, height: int, resolution: np.double, center_x: np.double, center_y: np.double,
                 max_iterations: int, use_dark_mode: bool, use_color: bool):
        self.width = width
        self.height = height

        self.resolution = resolution

        self.center_x = center_x
        self.center_y = center_y
        self.max_iterations = max_iterations
        self.use_dark_mode = use_dark_mode
        self.use_color = use_color

        self.mandelbrot_block_size = (16, 16, 1)
        self.mandelbrot_grid_size = (self.width // self.mandelbrot_block_size[0] + 1,
                                     self.height // self.mandelbrot_block_size[1] + 1, 1)

        self.colorize_block_size = (16, 16, 1)
        self.colorize_grid_size = (
            (self.width + self.colorize_block_size[0] - 1) // self.colorize_block_size[0],
            (self.height + self.colorize_block_size[1] - 1) // self.colorize_block_size[1]
        )

        self.mandelbrot_mod = None
        self.colorize_mod = None

        self.mandelbrot_gpu = None
        self.colorize_gpu = None

        self.mandelbrot_output_gpu = cuda.mem_alloc(width * height * 4)
        self.colored_mandelbrot_gpu = cuda.mem_alloc(3 * width * height)

        self.mandelbrot_kernel = """
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

        self.colorize_kernel = """
            __global__ void colorize(unsigned char *output, int *mandelbrot_data, int width, int height, int max_iterations, unsigned char *max_iterations_palette) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                if (idx < width && idy < height) {
                    int iterations = mandelbrot_data[idy * width + idx];
                    if (iterations == max_iterations) {
                        output[3 * (idy * width + idx)] = [COLORMODE];  // Red component
                        output[3 * (idy * width + idx) + 1] = [COLORMODE];  // Green component
                        output[3 * (idy * width + idx) + 2] = [COLORMODE];  // Blue component
                    } else {
                        output[3 * (idy * width + idx)] = iterations * 2;
                        output[3 * (idy * width + idx) + 1] = iterations * 4;
                        output[3 * (idy * width + idx) + 2] = iterations * 8;
                    }
                }
            }
        """

        self.__update_colorize_kernel()
        self.__build_kernels()

    def __update_colorize_kernel(self):
        self.colorize_kernel = self.colorize_kernel.replace("[COLORMODE]", "0") \
            if self.use_dark_mode else self.colorize_kernel.replace("[COLORMODE]", "255")

    def __build_kernels(self):
        self.mandelbrot_mod = SourceModule(self.mandelbrot_kernel)
        self.mandelbrot_gpu = self.mandelbrot_mod.get_function("mandelbrot")

        self.colorize_mod = SourceModule(self.colorize_kernel)
        self.colorize_gpu = self.colorize_mod.get_function("colorize")

    def run(self):
        self.mandelbrot_gpu(self.mandelbrot_output_gpu, np.int32(self.width), np.int32(self.height), self.x_min, self.x_max,
                       self.y_min, self.y_max, np.int32(self.max_iterations),
                       block=self.mandelbrot_block_size, grid=self.mandelbrot_grid_size)


def main():
    mandelbrot = Mandelbrot(1000, 1000, np.double(1.5), np.double(-.5), np.double(0),
                            int(1024*.02), True, False)


if __name__ == "__main__":
    main()
