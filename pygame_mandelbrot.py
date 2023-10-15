import pygame
import numpy as np
from Mandelbrot import Mandelbrot, Util
import pyperclip
import threading

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

img_width = LEFT_PANE_WIDTH
img_height = HEIGHT
center_x = np.double(-.5)
center_y = np.double(0)
previous_center_x = None
previous_center_y = None
resolution = np.double(1.5)
max_iterations = 255
normal_color = True
use_grayscale = False

resource_lock = threading.Lock()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

mandelbrot = Mandelbrot(img_width, img_height, resolution, center_x,
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
                            max_iterations, True, normal_color)
    mandelbrot.animate(center_x, center_y, resolution, 20, 100, False,
                       True, "Mandelbrot_export_animation")

    mandelbrot.free()


def save_large_image():
    global mandelbrot
    pygame.quit()
    print("Saving large image")
    mandelbrot.free()

    mandelbrot = Mandelbrot(16000, 16000, resolution, center_x, center_y,
                            max_iterations, True, normal_color)
    large_image = mandelbrot.generate()
    Util.save_image(large_image, "large_export")
    mandelbrot.free()


def update_values():
    mandelbrot.update_params(center_x, center_y, resolution, max_iterations, normal_color, img_width, img_height)


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

    width_text = font.render(f'Width: {img_width}', True, WHITE)
    screen.blit(width_text, (LEFT_PANE_WIDTH + 10, 250))

    height_text = font.render(f'Height: {img_height}', True, WHITE)
    screen.blit(height_text, (LEFT_PANE_WIDTH + 10, 290))


def save_file():
    with resource_lock:
        copy_text = (((copy_template.replace("[CENTER_X]", str(center_x)).
                       replace("[CENTER_Y]", str(center_y))).
                      replace("[RESOLUTION]", str(resolution))).
                     replace("[MAX_ITERATIONS]", str(max_iterations)))
        pyperclip.copy(copy_text)
        with open('config_export.txt', 'w') as f:
            f.write(copy_text)


def save_small_image(mandelbrot_image_to_save):
    save_img = np.flipud(mandelbrot_image_to_save)
    Util.save_image(np.rot90(save_img, 3), "small_export")


def move_to_clicked(mouse_x, mouse_y):
    global center_x, center_y
    global previous_center_x, previous_center_y
    with resource_lock:
        previous_center_x = center_x
        previous_center_y = center_y
        center_x, center_y = Util.calculate_center(
            mouse_x, mouse_y, LEFT_PANE_WIDTH, HEIGHT, resolution, center_x, center_y
        )


def restore_position():
    global center_x, center_y
    global previous_center_x, previous_center_y

    if previous_center_x is not None:
        with resource_lock:
            center_x = previous_center_x
            previous_center_x = None
            center_y = previous_center_y
            previous_center_y = None


speed_factor = 10.0
should_generate = False
running = True
clock = pygame.time.Clock()
mandelbrot_image = mandelbrot.generate()
mandelbrot_image = np.rot90(mandelbrot_image, 3)
prev_is_moving = False
frame_hold = 0
while running:
    delta_time = clock.tick(FPS) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if event.pos[0] < LEFT_PANE_WIDTH:
                    mouse_x, mouse_y = event.pos
                    move_thread = threading.Thread(target=move_to_clicked, args=(mouse_x, mouse_y,))
                    move_thread.start()

            if event.button == 6:
                restore_thread = threading.Thread(target=restore_position)
                restore_thread.start()

            if event.button == 4:
                with resource_lock:
                    resolution /= 1.1

            if event.button == 5:
                with resource_lock:
                    resolution *= 1.1

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_c:
                save_thread = threading.Thread(target=save_file)
                save_thread.start()

            if event.key == pygame.K_r:
                with resource_lock:
                    center_x = -.5
                    center_y = 0
                    resolution = 1.5
                    max_iterations = 255

            if event.key == pygame.K_f:
                normal_color = not normal_color

            if event.key == pygame.K_g:
                use_grayscale = not use_grayscale

            # Defined macros
            if event.key == pygame.K_1:
                with resource_lock:
                    center_x = -1.484585267
                    center_y = 0.0
                    resolution = 1e-08
                    max_iterations = 5120

            if event.key == pygame.K_2:
                with resource_lock:
                    resolution = 0.0001
                    center_x = -.717
                    center_y = -0.2498
                    max_iterations = 5120

            if event.key == pygame.K_3:
                with resource_lock:
                    center_x = -0.7173625
                    center_y = -0.2505295
                    resolution = 1e-05
                    max_iterations = 5120

            if event.key == pygame.K_4:
                with resource_lock:
                    center_x = -1.99998588123072
                    center_y = 0.0
                    resolution = 1e-14
                    max_iterations = 1023

            if event.key == pygame.K_5:
                with resource_lock:
                    center_x = -0.7765929020241705
                    center_y = -0.13664090727687
                    resolution = 1e-13
                    max_iterations = 2048

            if event.key == pygame.K_RIGHTBRACKET:
                with resource_lock:
                    max_iterations = 1024 * 5

            if event.key == pygame.K_v:
                save_img = mandelbrot_image.copy()
                image_save_thread = threading.Thread(target=save_small_image, args=(save_img,))
                image_save_thread.start()

            if event.key == pygame.K_b:
                save_large_image()

            if event.key == pygame.K_n:
                save_animation_to_here()

            if event.key == pygame.K_t:
                speed_factor = 10.0

            if event.key == pygame.K_y:
                speed_factor = 50.0

            if event.key == pygame.K_u:
                speed_factor = 3

    is_moving = False
    keys = pygame.key.get_pressed()
    if (keys[pygame.K_a] or keys[pygame.K_d] or keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_e] or
            keys[pygame.K_q] or keys[pygame.K_z] or keys[pygame.K_x]):
        is_moving = True
    if keys[pygame.K_a]:
        with resource_lock:
            center_x -= (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_d]:
        with resource_lock:
            center_x += (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_w]:
        with resource_lock:
            center_y -= (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_s]:
        with resource_lock:
            center_y += (0.1 * resolution) * speed_factor * delta_time

    if keys[pygame.K_e]:
        with resource_lock:
            resolution /= 1.1

    if keys[pygame.K_q]:
        with resource_lock:
            resolution *= 1.1

    if keys[pygame.K_z]:
        with resource_lock:
            max_iterations -= int(speed_factor * delta_time * 10)

    if keys[pygame.K_x]:
        with resource_lock:
            max_iterations += int(speed_factor * delta_time * 10)

    with resource_lock:
        center_x = np.clip(center_x, -2.0, 2.0)
        center_y = np.clip(center_y, -2.0, 2.0)
        resolution = np.clip(resolution, MIN_RESOLUTION, MAX_RESOLUTION)
        max_iterations = np.clip(max_iterations, 1, 1024 * 10)

    screen.fill(GRAY)

    if is_moving:
        img_width = LEFT_PANE_WIDTH // 2
        img_height = HEIGHT // 2
        should_generate = True
        prev_is_moving = is_moving
    else:
        if max_iterations > 5500 or resolution < 0.00000000001:
            img_width = int(LEFT_PANE_WIDTH // 1.5)
            img_height = int(HEIGHT // 1.5)
        else:
            img_width = LEFT_PANE_WIDTH
            img_height = HEIGHT

    if is_moving and not prev_is_moving:
        should_generate = True
    elif not is_moving and prev_is_moving and frame_hold < 3:
        should_generate = False
        frame_hold += 1
    elif not is_moving and prev_is_moving and frame_hold == 3:
        should_generate = True
        frame_hold = 0
        prev_is_moving = is_moving

    if should_generate:
        mandelbrot_image = mandelbrot.generate()
        mandelbrot_image = np.flipud(mandelbrot_image)
        mandelbrot_image = np.rot90(mandelbrot_image, 3)

    # TODO: Fix black and white mode
    surface = pygame.surfarray.make_surface(mandelbrot_image)
    if use_grayscale:
        surface = pygame.transform.grayscale(surface)
    scaled_mandelbrot_image = pygame.transform.scale(surface, (LEFT_PANE_WIDTH, HEIGHT))
    screen.blit(scaled_mandelbrot_image, (0, 0))

    pygame.draw.rect(screen, GRAY, (LEFT_PANE_WIDTH, 0, RIGHT_PANE_WIDTH, HEIGHT))

    update_static_elements(clock.get_fps(), delta_time)
    update_values()

    should_generate = not should_generate

    pygame.display.flip()

mandelbrot.free()
pygame.quit()
