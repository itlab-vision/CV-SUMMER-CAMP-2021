import math
import random

def get_random_colors(N):
    assert N > 0

    color_step = 10
    if (255//color_step) * (255//color_step) * (255//color_step) <= 2 * N:
        print("WARNING: Too many tracks to generate random colors with this color_step -- decrease color step")
        color_step = math.floor(math.pow(255*255*255//(2*N), 0.33333))
    color_step = int(color_step)
    assert (255//color_step) * (255//color_step) * (255//color_step) > 2*N

    colors = []
    set_shrinked_colors = set()
    while len(colors) < N:
        shrinked_B = random.randint(0, 255//color_step)
        shrinked_G = random.randint(0, 255//color_step)
        shrinked_R = random.randint(0, 255//color_step)

        shrinked_color = (shrinked_B, shrinked_G, shrinked_R)  # to be sure that the colors are sufficiently different
        if shrinked_color in set_shrinked_colors:
            continue

        color = (color_step*shrinked_B, color_step*shrinked_G, color_step*shrinked_R)
        colors.append(color)
    return colors
