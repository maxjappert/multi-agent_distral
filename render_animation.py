import numpy as np
from matplotlib import pyplot as plt

def create_plan_txt(p1_x, p1_y, p2_x, p2_y, goal=-1):
    """
    TARGET1 = D_GREEN = 2
    TARGET2 = L_GREEN = 3
    AGENT1 = D_RED = 4
    AGENT2 = L_RED = 5
    """

    grid_map = ["1 1 1 1 1 1 1 1 1 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 0 0 0 0 0 0 0 0 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 2 0 1 1 1 1 0 3 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 1 1 1 1 1 1 1 1 1"]

    grid_map_array = read_grid_map(grid_map)
    grid_map_array[p1_y, p1_x] = 4 if goal != 1 and goal != 2 else 6
    grid_map_array[p2_y, p2_x] = 5 if goal != 2 else 6

    return grid_map_array


def read_grid_map(grid_map):

    # Return the gridmap imported from a txt plan

    grid_map_array = []
    for k1 in grid_map:
        k1s = k1.split(' ')
        tmp_arr = []
        for k2 in k1s:
            try:
                tmp_arr.append(int(k2))
            except:
                pass
        grid_map_array.append(tmp_arr)

    grid_map_array = np.array(grid_map_array, dtype=int)
    return grid_map_array


counter = 0
def _gridmap_to_image(grid_map_array):

    # Return image from the gridmap

    # Used for the plan.txt files to define an environment
    EMPTY = BLACK = 0
    WALL = GRAY = 1
    TARGET1 = D_GREEN = 2
    TARGET2 = L_GREEN = 3
    AGENT1 = D_RED = 4
    AGENT2 = L_RED = 5
    SUCCESS = PINK = 6

    COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], D_GREEN: [0.0, 0.4, 0.0], L_GREEN: [0.5, 1.0, 0.5],
              D_RED: [0.4, 0.0, 0.0], L_RED: [1.0, 0.5, 0.5], PINK: [1.0, 0.0, 1.0]}

    img_shape = [256, 256, 3]
    observation = np.random.randn(*img_shape) * 0.0
    gs0 = int(observation.shape[0] / grid_map_array.shape[0])
    gs1 = int(observation.shape[1] / grid_map_array.shape[1])
    for i in range(grid_map_array.shape[0]):
        for j in range(grid_map_array.shape[1]):
            for k in range(3):
                this_value = COLORS[grid_map_array[i, j]][k]
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value

    img = (255 * observation).astype(np.uint8)
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    #plt.show()
    global counter
    counter += 1
    plt.savefig(f'images/anim{counter}')


_gridmap_to_image(create_plan_txt(2, 2, 7, 2))
_gridmap_to_image(create_plan_txt(2, 3, 7, 2))
_gridmap_to_image(create_plan_txt(2, 3, 7, 3))
_gridmap_to_image(create_plan_txt(2, 4, 7, 3))
_gridmap_to_image(create_plan_txt(3, 4, 7, 3))
_gridmap_to_image(create_plan_txt(4, 4, 7, 3))
_gridmap_to_image(create_plan_txt(5, 4, 7, 3))
_gridmap_to_image(create_plan_txt(6, 4, 7, 3))
_gridmap_to_image(create_plan_txt(7, 4, 7, 3))
_gridmap_to_image(create_plan_txt(8, 4, 7, 3))
_gridmap_to_image(create_plan_txt(8, 4, 7, 4))
_gridmap_to_image(create_plan_txt(8, 5, 7, 4))
_gridmap_to_image(create_plan_txt(8, 5, 6, 4))
_gridmap_to_image(create_plan_txt(8, 6, 6, 4, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 5, 4, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 4, 4, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 3, 4, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 2, 4, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 2, 5, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 2, 6, goal=1))
_gridmap_to_image(create_plan_txt(8, 6, 1, 6, goal=2))





