GRID_SIZE = (5, 5)


def test_location_to_coords():
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import GoalGrid

    space_min = np.array([-1, -1])
    space_max = np.array([1, 1])
    grid_width, grid_height = GRID_SIZE
    nb_pixels = 1000
    gg = GoalGrid(space_min, space_max, grid_width, grid_height)

    x = np.linspace(space_min[0], space_max[0], nb_pixels)
    y = np.linspace(space_min[1], space_max[1], nb_pixels)
    xx, yy = np.meshgrid(x, y)

    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    grid_coords = gg.cell_coords(coords).reshape(nb_pixels, nb_pixels, 2)
    image_from_coords = grid_coords[:, :, 0] + grid_coords[:, :, 1] * grid_height
    plt.imshow(image_from_coords)
    plt.show()


def test_generate_centers():
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import GoalGrid

    space_min = np.array([0, 0])
    space_max = np.array([1, 1])
    grid_width, grid_height = GRID_SIZE
    nb_pixels = 1000
    gg = GoalGrid(space_min, space_max, grid_width, grid_height)
    centers = gg.get_cells_center()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.scatter(centers[:, 0], gg.get_cells_center()[:, 1], vmin=0, vmax=1)
    ax.set_xticks(np.arange(0, 1.0, 1.0 / grid_width))
    ax.set_yticks(np.arange(0, 1.0, 1.0 / grid_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.show()


test_location_to_coords()
test_generate_centers()
