import math
import imageio
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import moviepy.editor as mp


class Util:
    """
    Utility class for static methods.
    """
    @staticmethod
    def save_image(array, name):
        """
        Saves the received numpy array, and opens the image.
        :param array: The generated numpy array.
        :param name: The name of the file.
        :return: None
        """
        mandelbrot_image = Image.fromarray(array)
        mandelbrot_image.save(name + ".png")
        mandelbrot_image.show()

    # Custom ease in out functions
    # Can define any function and pass it as parameter to the animation function
    @staticmethod
    def ease_in_out_quad(t):
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t

    @staticmethod
    def custom_ease_in_out(t):
        return (1 - math.cos(t * math.pi)) / 2


class Mandelbrot:
    """
    Generates mandelbrot set with cuda.
    Can generate images, and animations, with or without color.
    """
    def __init__(self, width: int, height: int, resolution: np.double, center_x: np.double, center_y: np.double,
                 max_iterations: int, use_dark_mode: bool, use_color: bool):
        """
        Initialize the generator.
        :param width: The width of the image in pixels.
        :param height: The height of the image in heights.
        :param resolution: The resolution of the generation. (Smaller number more zoomed in).
        :param center_x: The center of the view on the x (real) axis.
        :param center_y: The center of the view on the y (imaginary) axis.
        :param max_iterations: The maximum iterations for the function (contrast of the image).
        :param use_dark_mode: If true the parts with maximum iteration are black in colored mode instead of white.
        :param use_color: Weather or not to calculate color for pixels based on iteration number.
        """
        self.width = width
        self.height = height

        self.aspect = np.double(height / width)

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

        self.mandelbrot_output_gpu = cuda.mem_alloc(self.width * self.height * 4)
        self.colored_mandelbrot_gpu = cuda.mem_alloc(3 * self.width * self.height)

        self.mandelbrot_output_cpu = np.empty((self.height, self.width), dtype=np.int32)
        self.colored_mandelbrot_cpu = np.empty((self.height, self.width, 3), dtype=np.uint8)

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
        """
        Replaces the placeholder in the kernel source code.
        :return: None
        """
        self.colorize_kernel = self.colorize_kernel.replace("[COLORMODE]", "0") \
            if self.use_dark_mode else self.colorize_kernel.replace("[COLORMODE]", "255")

    def __build_kernels(self):
        """
        Builds the generation and coloring kernels for cuda.
        :return: None
        """
        self.mandelbrot_mod = SourceModule(self.mandelbrot_kernel)
        self.mandelbrot_gpu = self.mandelbrot_mod.get_function("mandelbrot")

        self.colorize_mod = SourceModule(self.colorize_kernel)
        self.colorize_gpu = self.colorize_mod.get_function("colorize")

    def __calculate_min_max(self, center_x, center_y, resolution):
        """
        Calculates the edges of the view based on the center coordinates and resolution..
        :param center_x: The center of the view on the x (real) axis.
        :param center_y: The center of the view on the y (imaginary) axis.
        :param resolution: The resolution of the generation. (Smaller number more zoomed in).
        :return: None
        """
        x_min = np.double(center_x - resolution)
        x_max = np.double(center_x + resolution)
        y_min = np.double(center_y - (resolution * self.aspect))
        y_max = np.double(center_y + (resolution * self.aspect))
        return x_min, x_max, y_min, y_max

    def __run(self, center_x, center_y, resolution):
        """
        Runs the kernel on the gpu that calculates the mandelbrot set.
        :param center_x: The center of the view on the x (real) axis.
        :param center_y: The center of the view on the y (imaginary) axis.
        :param resolution: The resolution of the generation. (Smaller number more zoomed in).
        :return: None
        """
        x_min, x_max, y_min, y_max = self.__calculate_min_max(center_x, center_y, resolution)

        self.mandelbrot_gpu(self.mandelbrot_output_gpu, np.int32(self.width), np.int32(self.height), x_min, x_max,
                            y_min, y_max, np.int32(self.max_iterations),
                            block=self.mandelbrot_block_size, grid=self.mandelbrot_grid_size)

    def __run_color(self):
        """
        Runs the coloring kernel on the gpu.
        :return: None
        """
        self.colorize_gpu(self.colored_mandelbrot_gpu, self.mandelbrot_output_gpu,
                          np.int32(self.width), np.int32(self.height), np.int32(self.max_iterations),
                          block=self.colorize_block_size, grid=self.colorize_grid_size)

    def __reset(self):
        """
        Resets the output arrays.
        :return: None
        """
        self.mandelbrot_output_cpu = np.empty((self.height, self.width), dtype=np.int32)
        self.colored_mandelbrot_cpu = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def free(self):
        """
        When done free the gpu memory arrays.
        !!! If used, needs to be reinitialized !!!
        :return: None
        """
        self.mandelbrot_output_gpu.free()
        self.colored_mandelbrot_gpu.free()

    def __copy_back_cpu(self):
        """
        Copies the data from the gpu to the cpu, and normalizes the output.
        :return: None
        """
        cuda.memcpy_dtoh(self.mandelbrot_output_cpu, self.mandelbrot_output_gpu)

        output_normalized = (255 * (self.mandelbrot_output_cpu - np.min(self.mandelbrot_output_cpu)) /
                             np.ptp(self.mandelbrot_output_cpu)).astype(np.uint8)

        self.__reset()

        return output_normalized

    def __copy_back_color_cpu(self):
        """
        Copies the data from the coloring process from the gpu to the cpu.
        :return: None
        """
        cuda.memcpy_dtoh(self.colored_mandelbrot_cpu, self.colored_mandelbrot_gpu)
        colored_copy = self.colored_mandelbrot_cpu.copy()

        self.__reset()

        return colored_copy

    def generate(self):
        """
        Generates the image array using the initialized settings.
        :return: The numpy array of the image.
        """
        self.__run(self.center_x, self.center_y, self.resolution)

        if not self.use_color:
            return self.__copy_back_cpu()
        else:
            self.__run_color()
            return self.__copy_back_color_cpu()

    def __generate(self, current_center_x, current_center_y, current_resolution):
        """
        Generates the image array using the provided settings.
        Internally used for animation.
        :param current_center_x: The center of the view on the x (real) axis.
        :param current_center_y: The center of the view on the y (imaginary) axis.
        :param current_resolution: The resolution of the generation. (Smaller number more zoomed in).
        :return: The numpy array of the image.
        """
        self.__run(current_center_x, current_center_y, current_resolution)

        if not self.use_color:
            return self.__copy_back_cpu()
        else:
            self.__run_color()
            return self.__copy_back_color_cpu()

    def update_params(self, new_center_x: np.double, new_center_y: np.double,
                      new_resolution: np.double, new_max_iter: int):
        """
        Update the settings without having to reinitialize.
        :param new_center_x: The center of the view on the x (real) axis.
        :param new_center_y: The center of the view on the y (imaginary) axis.
        :param new_resolution: The resolution of the generation. (Smaller number more zoomed in).
        :param new_max_iter: The maximum iterations for the function (contrast of the image).
        :return: None
        """
        self.__reset()
        self.center_x = new_center_x
        self.center_y = new_center_y
        self.resolution = new_resolution
        self.max_iterations = new_max_iter

    def __calculate_increments(self, target_center_x, target_center_y, target_resolution,
                               target_frames, move_percent, transition_function, starting_resolution,
                               first_part_resolution):
        """
        Calculate the increments for the animation.
        :param target_center_x: The center of the target view on the x (real) axis.
        :param target_center_y: The center of the target view on the y (imaginary) axis.
        :param target_resolution: The resolution of the target generation. (Smaller number more zoomed in).
        :param target_frames: The number of frames to generate for.
        :param move_percent: The percentage of the frames used for aligning the x-axis and y-axis.
        :param transition_function: The function for easing the movement. Can be user defined.
        :param starting_resolution: The resolution to start at for the animation.
        :param first_part_resolution: The resolution to go to while aligning the x-axis and y-axis.
        :return: The increments array for center_x, center_y, and resolution
        """
        percent_of_frames = int(target_frames * move_percent)

        easing = np.vectorize(transition_function)
        t = np.linspace(0, 1, percent_of_frames)

        center_x_increments = np.ones(target_frames) * target_center_x
        center_x_increments[0:percent_of_frames] = (1 - easing(t)) * self.center_x + easing(t) * target_center_x

        center_y_increments = np.ones(target_frames) * target_center_y
        center_y_increments[0:percent_of_frames] = (1 - easing(t)) * self.center_y + easing(t) * target_center_y

        resolution_increments = np.ones(target_frames) * target_resolution
        resolution_increments[0:percent_of_frames] = np.geomspace(starting_resolution, first_part_resolution,
                                                                  percent_of_frames)
        resolution_increments[percent_of_frames:] = np.geomspace(first_part_resolution, target_resolution,
                                                                 target_frames - percent_of_frames)

        return center_x_increments, center_y_increments, resolution_increments

    def animate(self, target_center_x, target_center_y, target_resolution, fps,
                target_frames, loop_animation=False, convert_to_webm=False, name="mandelbrot_animation",
                webm_bitrate="30000k", move_frame_percent=0.1, transition_method=Util.ease_in_out_quad,
                starting_resolution=2.0, first_part_resolution=1.0):
        """
        Animate a gif or webm that zooms in to the set.
        :param target_center_x: The center of the target view on the x (real) axis.
        :param target_center_y: The center of the target view on the y (imaginary) axis.
        :param target_resolution: The resolution of the target generation. (Smaller number more zoomed in).
        :param fps: The framerate of the gif / webm.
        :param target_frames: The number of frames to generate for.
        :param loop_animation: Weather or not to reverse the animation and append the reversed frames to the end.
        :param convert_to_webm: Weather or not to convert the gif to a webm.
        :param name: The name of the file to save.
        :param webm_bitrate: The bitrate of the webm file.
        :param move_frame_percent: The percentage of the frames used for aligning the x-axis and y-axis.
        :param transition_method: The function for easing the movement. Can be user defined.
        :param starting_resolution: The resolution to start at for the animation.
        :param first_part_resolution: The resolution to go to while aligning the x-axis and y-axis.
        :return: None
        """
        frames = []

        # Calculate the increments using linear interpolation
        center_x_increments, center_y_increments, resolution_increments = self.__calculate_increments(
            target_center_x, target_center_y, target_resolution, target_frames, move_frame_percent, transition_method,
            starting_resolution, first_part_resolution)

        for i in range(target_frames):
            print("starting frame " + str(i))

            # Get the current values for center and resolution
            current_center_x = center_x_increments[i]
            current_center_y = center_y_increments[i]
            current_resolution = resolution_increments[i]

            # Calculate the Mandelbrot set for the current frame
            result = self.__generate(current_center_x, current_center_y, current_resolution)

            # Convert the current frame to an image and append it to the frames list
            mandelbrot_image = Image.fromarray(result)
            frames.append(mandelbrot_image)

        if loop_animation:
            frames_reverse = frames.copy()
            # First and last images are the same
            frames_reverse.pop()
            frames_reverse.reverse()
            # First and last images are the same
            frames_reverse.pop()

            frames = frames + frames_reverse

        print("Converting frames to gif")
        imageio.mimsave(name + ".gif", frames, loop=4, fps=fps)
        if convert_to_webm:
            print("Converting gif to webm")
            clip = mp.VideoFileClip(name + ".gif")
            clip.write_videofile(name + ".webm", bitrate=webm_bitrate, fps=fps)


def main():
    mandelbrot = Mandelbrot(800, 600, np.double(2), np.double(-.5), np.double(0),
                            int(1024*10), False, True)
    # mandelbrot = Mandelbrot(800, 600, np.double(0.000000000000001), np.double(-1.99998588123072), np.double(0),
    #                         int(1024 * 10), True, True)

    mandelbrot.animate(np.double(-1.99998588123072), 0, 0.000000000000001,
                       20, 100, move_frame_percent=0.2, transition_method=Util.custom_ease_in_out)

    # array = mandelbrot.generate()
    # Util.save_image(array, "test")
    # mandelbrot.update_params(np.double(-2.0), np.double(1.0), np.double(1.5), 1024*10)
    # array = mandelbrot.generate()
    # Util.save_image(array, "test2")
    mandelbrot.free()


if __name__ == "__main__":
    main()
