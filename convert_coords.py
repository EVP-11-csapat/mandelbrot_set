import sys
import yaml

img_x = int(sys.argv[1])
img_y = int(sys.argv[2])

with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

width = int(config["width"])
height = (config["height"])
aspect = (height / width)
resolution = float(config["resolution"])
center_x = float(config["center_x"])
center_y = float(config["center_y"])

left = center_x - resolution
bottom = center_y - (resolution * aspect)

translate_x = (img_x / width) * (resolution * 2)
translate_y = (img_y / width) * (resolution * 2)

print('center_x : ', translate_x + left)
print('center_y : ', translate_y + bottom)
