## Mandelbrot Set Generator

This project implements a Mandelbrot set image generator using PyCUDA. PyCUDA is a Python library that allows you to use NVIDIA GPUs to accelerate your code. The Mandelbrot set is a beautiful and complex mathematical fractal that can be visualized by plotting the points in the complex plane that do not diverge to infinity under repeated iteration of the Mandelbrot function:

$$f_{c}(z)=z^{2}+c$$

## Usage

To generate an image of the Mandelbrot set, run the following command:

```bash
python main.py
```

This will generate an image named `mandelbrot_set.png` in the current directory.

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

# Pygame Mandelbrot Set Viewer

This is a simple Mandelbrot Set viewer created using Pygame. You can explore the Mandelbrot Set using the following keybindings:

## Mouse Controls

- **Left Click:** Move to the clicked position within the left pane.
- **Scroll Up:** Zoom in (decrease resolution).
- **Scroll Down:** Zoom out (increase resolution).
- **Back Button (on mouse):** Restore the previous position.

## Keybindings

### General Controls

- **Esc:** Quit the application.

### Copy Configuration

- **C:** Save the current configuration in a separate thread.

### Reset Configuration

- **R:** Reset to the default configuration, which sets:
  - Center X: -0.5
  - Center Y: 0.0
  - Resolution: 1.5
  - Max Iterations: 255

### Toggle Color

- **F:** Toggle color visualization on or off.

### Predefined Macros

You can jump to predefined configurations using these keys:

- **1:** Predefined Configuration 1
  - Center X: -1.484585267
  - Center Y: 0.0
  - Resolution: 1e-08
  - Max Iterations: 5120

- **2:** Predefined Configuration 2
  - Center X: -0.717
  - Center Y: -0.2498
  - Resolution: 0.0001
  - Max Iterations: 5120

- **3:** Predefined Configuration 3
  - Center X: -0.7173625
  - Center Y: -0.2505295
  - Resolution: 1e-05
  - Max Iterations: 5120

- **4:** Predefined Configuration 4
  - Center X: -1.99998588123072
  - Center Y: 0.0
  - Resolution: 1e-14
  - Max Iterations: 1023

- **5:** Predefined Configuration 5
  - Center X: -0.7765929020241705
  - Center Y: -0.13664090727687
  - Resolution: 1e-13
  - Max Iterations: 2048

### Iteration Adjustment

- **]:** Increase the maximum number of iterations (Max Iterations) fivefold.

### Save Images

- **V:** Save a small rotated image in a separate thread.

- **B:** Save a large image.

### Animation

- **N:** Save the animation to the current frame.

### Speed Adjustment

- **U:** Set speed factor to 3.0 (ultra slow)

- **T:** Set speed factor to 10.0 (slow).

- **Y:** Set speed factor to 50.0 (fast).

### Navigation

You can navigate through the Mandelbrot set using the following keys:

- **A:** Move left.

- **D:** Move right.

- **W:** Move up.

- **S:** Move down.

### Zoom

- **E:** Zoom in (decrease resolution).

- **Q:** Zoom out (increase resolution).

### Iteration Adjustment

- **Z:** Decrease the maximum number of iterations per frame.

- **X:** Increase the maximum number of iterations per frame.

## Configuration Limits

- The center X and center Y are limited to the range [-2.0, 2.0].

- Resolution is limited within a predefined range (MIN_RESOLUTION to MAX_RESOLUTION).

- Maximum iterations are limited to a maximum of 10,240.

## Enjoy exploring the Mandelbrot Set!