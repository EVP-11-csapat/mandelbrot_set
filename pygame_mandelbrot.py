import pygame
import numpy as np
from Mandelbrot import Mandelbrot, Util
import pyperclip

pygame.init()

WIDTH = 1200
HEIGHT = 800
LEFT_PANE_WIDTH = 800
RIGHT_PANE_WIDTH = 400
FPS = 30
MIN_RESOLUTION = 0.00000000000001
MAX_RESOLUTION = 2.0

GRAY = (100, 100, 100)
WHITE = (255, 255, 255)

center_x = np.double(-.5)
center_y = np.double(0)
resolution = np.double(1.5)
max_iterations = 255
use_color = True

screen = pygame.display.set_mode((WIDTH, HEIGHT))

mandelbrot = Mandelbrot(LEFT_PANE_WIDTH, HEIGHT, resolution, center_x,
                        center_y, max_iterations, False, True)

font = pygame.font.Font(None, 24)

copy_template = """
center_x = [CENTER_X]
center_y = [CENTER_Y]
resolution = [RESOLUTION]
max_iterations = [MAX_ITERATIONS]
"""


def save_animation_to_here():
    global mandelbrot
    pygame.quit()
    print("Saving animation")
    mandelbrot.free()

    mandelbrot = Mandelbrot(1000, 1000, np.double(2), np.double(-.5), np.double(0),
                            max_iterations, True, use_color)
    mandelbrot.animate(center_x, center_y, resolution, 20, 100, False,
                       True, "Mandelbrot_export_animation")

    mandelbrot.free()

def save_large_image():
    global mandelbrot
    pygame.quit()
    print("Saving large image")
    mandelbrot.free()

    mandelbrot = Mandelbrot(16000, 16000, resolution, center_x, center_y,
                            max_iterations, True, use_color)
    large_image = mandelbrot.generate()
    Util.save_image(large_image, "large_export")
    mandelbrot.free()


def update_values():
    mandelbrot.update_params(center_x, center_y, resolution, max_iterations, use_color)


def update_static_elements(fps, delta_time):
    fps_text = font.render(f'FPS: {fps}', True, WHITE)
    screen.blit(fps_text, (LEFT_PANE_WIDTH + 10, 10))

    delta_time_text = font.render(f'Delta Time: {delta_time} s', True, WHITE)
    screen.blit(delta_time_text, (LEFT_PANE_WIDTH + 10, 50))

    center_x_text = font.render(f'Center X: {center_x}', True, WHITE)
    screen.blit(center_x_text, (LEFT_PANE_WIDTH + 10, 90))

    center_y_text = font.render(f'Center Y: {center_y}', True, WHITE)
    screen.blit(center_y_text, (LEFT_PANE_WIDTH + 10, 130))

    resolution_text = font.render(f'Resolution: {resolution}', True, WHITE)
    screen.blit(resolution_text, (LEFT_PANE_WIDTH + 10, 170))

    max_iterations_text = font.render(f'Max Iterations: {max_iterations}', True, WHITE)
    screen.blit(max_iterations_text, (LEFT_PANE_WIDTH + 10, 210))


speed_factor = 10.0
should_generate = False
running = True
clock = pygame.time.Clock()
mandelbrot_image = mandelbrot.generate()
mandelbrot_image = np.rot90(mandelbrot_image, 3)
while running:
    delta_time = clock.tick(FPS) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_c:
                copy_text = (((copy_template.replace("[CENTER_X]", str(center_x)).
                             replace("[CENTER_Y]", str(center_y))).
                             replace("[RESOLUTION]", str(resolution))).
                             replace("[MAX_ITERATIONS]", str(max_iterations)))
                pyperclip.copy(copy_text)

            if event.key == pygame.K_r:
                center_x = -.5
                center_y = 0
                resolution = 1.5
                max_iterations = 255

            if event.key == pygame.K_f:
                use_color = not use_color

            # Defined macros
            if event.key == pygame.K_1:
                center_x = -1.484585267
                center_y = 0.0
                resolution = 1e-08
                max_iterations = 5120

            if event.key == pygame.K_2:
                resolution = 0.0001
                center_x = -.717
                center_y = -0.2498
                max_iterations = 5120

            if event.key == pygame.K_3:
                center_x = -0.7173625
                center_y = -0.2505295
                resolution = 1e-05
                max_iterations = 5120

            if event.key == pygame.K_4:
                center_x = -1.99998588123072
                center_y = 0.0
                resolution = 1e-14
                max_iterations = 1023

            if event.key == pygame.K_5:
                center_x = -0.7765929020241705
                center_y = -0.13664090727687
                resolution = 1e-13
                max_iterations = 2048

            if event.key == pygame.K_RIGHTBRACKET:
                max_iterations = 1024*5

            if event.key == pygame.K_v:
                save_img = mandelbrot_image.copy()
                save_img = np.flipud(save_img)
                Util.save_image(np.rot90(save_img, 3), "small_export")

            if event.key == pygame.K_b:
                save_large_image()

            if event.key == pygame.K_n:
                save_animation_to_here()

            if event.key == pygame.K_t:
                speed_factor = 10.0

            if event.key == pygame.K_y:
                speed_factor = 50.0

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        center_x -= (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_d]:
        center_x += (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_w]:
        center_y -= (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_s]:
        center_y += (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_e]:
        resolution /= 1.1

    if keys[pygame.K_q]:
        resolution *= 1.1

    if keys[pygame.K_z]:
        max_iterations -= int(speed_factor * delta_time * 10)

    if keys[pygame.K_x]:
        max_iterations += int(speed_factor * delta_time * 10)

    center_x = np.clip(center_x, -2.0, 2.0)
    center_y = np.clip(center_y, -2.0, 2.0)
    resolution = np.clip(resolution, MIN_RESOLUTION, MAX_RESOLUTION)
    max_iterations = np.clip(max_iterations, 1, 1024*10)

    screen.fill(GRAY)

    if should_generate:
        mandelbrot_image = mandelbrot.generate()
        mandelbrot_image = np.flipud(mandelbrot_image)
        mandelbrot_image = np.rot90(mandelbrot_image, 3)

    # TODO: Fix black and white mode
    mandelbrot_surface = pygame.surfarray.make_surface(mandelbrot_image)
    screen.blit(mandelbrot_surface, (0, 0))

    pygame.draw.rect(screen, GRAY, (LEFT_PANE_WIDTH, 0, RIGHT_PANE_WIDTH, HEIGHT))

    update_static_elements(clock.get_fps(), delta_time)
    update_values()

    should_generate = not should_generate

    pygame.display.flip()

mandelbrot.free()
pygame.quit()