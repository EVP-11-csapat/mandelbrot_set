# import pygame
# import pygame_gui
# from PIL import Image
# import mandelbrot_util
#
# pygame.init()
# mandelbrot_util.init()
#
# pygame.display.set_caption("Mandelbrot Viewer")
# window_surface = pygame.display.set_mode((800, 600))
#
# background = pygame.Surface((800, 600))
# background.fill(pygame.Color('#000000'))
#
# manager = pygame_gui.UIManager((800, 600))
#
# image = None
#
# hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 275), (100, 50)),
#                                             text='Say Hello', manager=manager)
#
#
# def print_to_screen():
#     global background
#     background = pygame.image.load("mandelbrot_setUIGENERATED.png")
#
#
# def make_mandelbrot_gray():
#     global image
#     print("Calling make_mandelbrot_gray")
#     mandelbrot_array = mandelbrot_util.make_mandelbrot(1000, 1000, -2.0, 1.0, -1.5, 1.5, 255)
#     image = Image.fromarray(mandelbrot_array)
#     image.save("mandelbrot_setUIGENERATED.png")
#
#     print_to_screen()
#
#
#
# clock = pygame.time.Clock()
# is_running = True
# while is_running:
#     time_delta = clock.tick(60) / 1000.0
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             is_running = False
#
#         if event.type == pygame_gui.UI_BUTTON_PRESSED:
#             if event.ui_element == hello_button:
#                 make_mandelbrot_gray()
#
#         manager.process_events(event)
#
#     manager.update(time_delta)
#
#     window_surface.blit(background, (0, 0))
#     manager.draw_ui(window_surface)
#
#     pygame.display.update()
#
# pygame.quit()
#########################################
# import pygame
# import pygame_gui
#
# pygame.init()
#
# WIDTH, HEIGHT = 800, 800
# LEFT_PANE_WIDTH, LEFT_PANE_HEIGHT = 600, 800
# RIGHT_PANE_WIDTH, RIGHT_PANE_HEIGHT = 200, 800
#
# window = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Mandelbrot Set Explorer")
#
# gui_manager = pygame_gui.UIManager((RIGHT_PANE_WIDTH, RIGHT_PANE_HEIGHT))
#
# left_pane = pygame.Surface((LEFT_PANE_WIDTH, LEFT_PANE_HEIGHT))
# left_pane.fill((0, 0, 0))
#
#
# def load_image(image_path):
#     image = pygame.image.load(image_path)
#     left_pane.blit(image, (0, 0))
#
#
# right_pane = pygame_gui.elements.UIPanel(
#     relative_rect=pygame.Rect((600, 0, RIGHT_PANE_WIDTH, RIGHT_PANE_HEIGHT)),
#     starting_layer_height=1,
#     manager=gui_manager
# )
#
# generate_button = pygame_gui.elements.UIButton(
#     relative_rect=pygame.Rect((RIGHT_PANE_WIDTH // 2 - 75, RIGHT_PANE_HEIGHT // 2 - 25, 150, 50)),
#     text="Generate",
#     manager=gui_manager
# )
#
# running = True
# clock = pygame.time.Clock()
# while running:
#     delta_time = clock.tick(60) / 1000.0
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#
#         gui_manager.process_events(event)
#
#     gui_manager.update(delta_time)
#
#     window.fill((255, 255, 255))
#
#     window.blit(left_pane, (0, 0))
#     gui_manager.draw_ui(window)
#
#     pygame.display.update()
#
# pygame.quit()

import pygame
import pygame_gui
import mandelbrot_util
from PIL import Image

mandelbrot_util.init()

# Initialize Pygame
pygame.init()

# Define the window dimensions
window_width = 800
window_height = 600

# Create the Pygame window
screen = pygame.display.set_mode((window_width, window_height))

# Set the title of the window
pygame.display.set_caption("Two Panes with Button")

# Create a UIManager for pygame_gui
ui_manager = pygame_gui.UIManager((window_width, window_height))

# Define the colors
black = (0, 0, 0)
gray = (128, 128, 128)

# Create the left pane (black background)
left_pane = pygame.Rect(0, 0, 600, 800)

# Create the right pane (gray background)
right_pane = pygame.Rect(600, 0, 200, 800)

# Create a button in the right pane
button_rect = pygame.Rect(620, 20, 160, 40)
button = pygame_gui.elements.UIButton(relative_rect=button_rect, text="Click Me", manager=ui_manager)


# Method for updating the screen
def update_screen(image_path=None):
    screen.fill(black)
    pygame.draw.rect(screen, gray, right_pane)

    if not image_path:
        pygame.draw.rect(screen, black, left_pane)
    else:
        image = pygame.image.load(image_path)

        # Get the dimensions of the left pane
        left_pane_width = left_pane.width
        left_pane_height = left_pane.height

        # Get the dimensions of the loaded image
        image_width = image.get_width()
        image_height = image.get_height()

        # Calculate the scaling factors for width and height to fit the image in the left pane
        width_scale = left_pane_width / image_width
        height_scale = left_pane_height / image_height

        # Use the smaller scaling factor to preserve aspect ratio
        min_scale = min(width_scale, height_scale)

        # Scale the image
        new_width = int(image_width * min_scale)
        new_height = int(image_height * min_scale)
        image = pygame.transform.scale(image, (new_width, new_height))

        # Calculate the position to center the scaled image in the left pane
        x = left_pane.left + (left_pane_width - new_width) // 2

        # If the image is smaller in height, start it at the top
        if new_height < left_pane_height:
            y = left_pane.top
        else:
            y = left_pane.top + (left_pane_height - new_height) // 2

        # Display the scaled image
        screen.blit(image, (x, y))


def generate():
    print("Calling make_mandelbrot_gray")
    mandelbrot_array = mandelbrot_util.make_mandelbrot(1000, 1000, -2.0, 1.0, -1.5, 1.5, 255)
    image = Image.fromarray(mandelbrot_array)
    image.save("mandelbrot_setUIGENERATED.png")
    global image_path
    image_path = "mandelbrot_setUIGENERATED.png"


# Main game loop
running = True
clock = pygame.time.Clock()
image_path = None
while running:
    delta_time = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        ui_manager.process_events(event)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == button:
                generate()

    # Update the screen using the method

    update_screen(image_path)

    ui_manager.update(delta_time)
    ui_manager.draw_ui(screen)

    pygame.display.update()


# Quit Pygame
pygame.quit()
