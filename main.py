import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
import yaml
import imageio
import moviepy.editor as mp

# User Config
with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

print(config)

use_dark_mode = bool(config["use_dark_mode"])
width = int(config["width"])
height = (config["height"])
try:
    max_iterations = int(config["max_iterations"])
except ValueError:
    try:
        max_iterations = int(eval(config["max_iterations"]))
    except SyntaxError:
        print("Max iterations config incorrect")
        exit(-1)

resolution = float(config["resolution"])
center_x = float(config["center_x"])
center_y = float(config["center_y"])
use_profile = int(config["use_profile"])
use_color = bool(config["use_color"])
do_animation = bool(config["do_animation"])

if do_animation:
    use_color = True
    target_center_x = config["target_center_x"]
    target_center_y = config["target_center_y"]
    target_resolution = config["target_resolution"]
    target_frames = config["target_frames"]
    target_frame_rate = config["target_frame_rate"]
    should_make_webm = bool(config["should_make_webm"])
    webm_bitrate = config["webm_bitrate"]
    resolution = config["starting_resolution"]

print(f"""
Using the following configuration:
Using dark-mode: {use_dark_mode}
Using profile: {use_profile}
Doing animation: {do_animation}
Using color: {use_color}
Image width: {width}
Image height: {height}
Image center X: {center_x}
Image center Y: {center_y}
Image resolution: {resolution}
""")


if use_profile == 0:
    max_iterations = 1024 if use_color else 128
    resolution = 1.5
    center_x = -.5
    center_y = 0
elif use_profile == 1:
    max_iterations = 2046*2 if use_color else 1024*2
    resolution = 0.00000001
    center_x = -1.484585267
    center_y = 0
elif use_profile == 2:
    max_iterations = 1024*5 if use_color else 1024
    resolution = 0.0001
    center_x = -.717
    center_y = -0.2498
elif use_profile == 3:
    max_iterations = 1024*5 if use_color else 1024
    resolution = 0.00001
    center_x = -.7173625
    center_y = -0.2505295
elif use_profile == 4:
    max_iterations = 1024*10
    resolution = 0.00000000000001
    center_x = -1.99998588123072
    center_y = 0
elif use_profile == 5:
    max_iterations = 1024*10
    resolution = 0.0000000000001
    center_x = -0.7765929020241705
    center_y = -0.13664090727687


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
if not do_animation:
    mandelbrot_gpu(output_gpu, np.int32(width), np.int32(height), np.double(x_min), np.double(x_max),
                   np.double(y_min), np.double(y_max), np.int32(max_iterations),
                   block=block_size, grid=grid_size)

colorize_kernel = """
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
            // Use the same color palette as before
            output[3 * (idy * width + idx)] = iterations * 2;
            output[3 * (idy * width + idx) + 1] = iterations * 4;
            output[3 * (idy * width + idx) + 2] = iterations * 8;
        }
    }
}
"""

# Compile the colorization CUDA kernel
colorize_mod = SourceModule(colorize_kernel.replace("[COLORMODE]", "0") if use_dark_mode else colorize_kernel.replace("[COLORMODE]", "255"))

# Get a reference to the CUDA kernel function
colorize_gpu = colorize_mod.get_function("colorize")

# Create a GPU array to store the colorized image
colored_mandelbrot_gpu = cuda.mem_alloc(3 * width * height)

# Define thread block and grid dimensions for colorization
colorize_block_size = (16, 16, 1)
colorize_grid_size = (
    (width + colorize_block_size[0] - 1) // colorize_block_size[0],
    (height + colorize_block_size[1] - 1) // colorize_block_size[1]
)

# Colorize the Mandelbrot set on the GPU
if use_color:
    if not do_animation:
        colorize_gpu(colored_mandelbrot_gpu, output_gpu, np.int32(width), np.int32(height), np.int32(max_iterations),
                     block=colorize_block_size, grid=colorize_grid_size)

        # Copy the colored image back to the CPU
        colored_mandelbrot_cpu = np.empty((height, width, 3), dtype=np.uint8)
        cuda.memcpy_dtoh(colored_mandelbrot_cpu, colored_mandelbrot_gpu)
    else:
        frames = []

        # Define the initial and target values for center_x, center_y, and resolution
        initial_center_x = center_x
        initial_center_y = center_y
        initial_resolution = resolution

        target_center_x = target_center_x
        target_center_y = target_center_y
        target_resolution = target_resolution

        # Calculate the increments using linear interpolation
        center_x_increments = np.ones(target_frames) * target_center_x
        center_x_increments[0:target_frames//6] = np.linspace(initial_center_x, target_center_x, target_frames // 6)
        center_y_increments = np.linspace(initial_center_y, target_center_y, target_frames)
        resolution_increments = np.geomspace(initial_resolution, target_resolution, target_frames)

        for i in range(target_frames):
            print("starting frame " + str(i))

            # Get the current values for center and resolution
            current_center_x = center_x_increments[i]
            current_center_y = center_y_increments[i]
            current_resolution = resolution_increments[i]

            # Modify x_min, x_max, y_min, and y_max using the current values
            x_min = current_center_x - current_resolution
            x_max = current_center_x + current_resolution
            y_min = current_center_y - (current_resolution * aspect)
            y_max = current_center_y + (current_resolution * aspect)

            # Calculate the Mandelbrot set for the current frame
            mandelbrot_gpu(output_gpu, np.int32(width), np.int32(height), np.double(x_min), np.double(x_max),
                           np.double(y_min), np.double(y_max), np.int32(max_iterations),
                           block=block_size, grid=grid_size)

            # Colorize the Mandelbrot set for the current frame
            colorize_gpu(colored_mandelbrot_gpu, output_gpu, np.int32(width), np.int32(height),
                         np.int32(max_iterations),
                         block=colorize_block_size, grid=colorize_grid_size)

            # Copy the colored image back to the CPU
            colored_mandelbrot_cpu = np.empty((height, width, 3), dtype=np.uint8)
            cuda.memcpy_dtoh(colored_mandelbrot_cpu, colored_mandelbrot_gpu)

            # Convert the current frame to an image and append it to the frames list
            mandelbrot_image = Image.fromarray(colored_mandelbrot_cpu)
            frames.append(mandelbrot_image)

        print(len(frames))
        frames_reverse = frames.copy()
        frames_reverse.pop()
        frames_reverse.reverse()
        frames_reverse.pop()

        frames = frames + frames_reverse

        print("Converting frames to gif")
        imageio.mimsave("mandelbrot_animation.gif", frames, loop=4, fps=target_frame_rate)
        if should_make_webm:
            print("Converting gif to webm")
            clip = mp.VideoFileClip("mandelbrot_animation.gif")
            clip.write_videofile("mandelbrot_animation.webm", bitrate=webm_bitrate, fps=target_frame_rate)
else:
    # Copy the result back to the CPU
    output_cpu = np.empty((height, width), dtype=np.int32)
    cuda.memcpy_dtoh(output_cpu, output_gpu)

    # Normalize the Mandelbrot set data to the range [0, 255]
    output_normalized = (255 * (output_cpu - np.min(output_cpu)) / np.ptp(output_cpu)).astype(np.uint8)

if not do_animation:
    # Convert the colored Mandelbrot set data to an image
    mandelbrot_image = Image.fromarray(colored_mandelbrot_cpu if use_color else output_normalized)

    # Save the image to a file
    mandelbrot_image.save("mandelbrot_set.png")

    # Display the image
    mandelbrot_image.show()

output_gpu.free()
