## Mandelbrot Set Generator

This project implements a Mandelbrot set image generator using PyCUDA. PyCUDA is a Python library that allows you to use NVIDIA GPUs to accelerate your code. The Mandelbrot set is a beautiful and complex mathematical fractal that can be visualized by plotting the points in the complex plane that do not diverge to infinity under repeated iteration of the Mandelbrot function:

$$f_{c}(z)=z^{2}+c$$

## Usage

To generate an image of the Mandelbrot set, run the following command:

```bash
python main.py
```

This will generate an image named `mandelbrot.png` in the current directory.

## Configuration

The configuration of the image is controlled by the `config.yaml` file. The following options are available:

* `use_dark_mode`: Whether to use a dark mode or not.
* `width`: The width of the image in pixels.
* `height`: The height of the image in pixels.
* `max_iterations`: The maximum number of iterations to use when calculating the Mandelbrot set.
* `resolution`: The resolution of the image.
* `center_x`: The center of the viewport on the real axis.
* `center_y`: The center of the viewport on the imaginary axis.
* `use_color`: Whether to use color or not.
* `use_profile`: Whether to use a predefined profile or not.

## Profiles

The following profiles are available:

* `DEFAULT`: The default profile.
* `SMALL`: A smaller version of the image.
* `SPIRALS`: A version of the image that focuses on the spirals.
