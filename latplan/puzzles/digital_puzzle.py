#!/usr/bin/env python3

import numpy as np
from .puzzle import generate_configs, successors

def generate(configs, width, height):
    assert width*height <= 16
    base_width = 5
    base_height = 6
    dim_x = base_width*width
    dim_y = base_height*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for digit,pos in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*base_height:(y+1)*base_height,
                   x*base_width:(x+1)*base_width] = panels[digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

def states(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    return generate(configs,width,height)

def transitions(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    transitions = np.array([ generate([c1,c2],width,height)
                             for c1 in configs for c2 in successors(c1,width,height) ])
    return np.einsum('ab...->ba...',transitions)

panels = [
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 0, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 0, 0, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],],
]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_image(a,name):
        plt.figure(figsize=(6,6))
        plt.imshow(a,interpolation='nearest',cmap='gray',)
        plt.savefig(name)
    def plot_grid(images,name="plan.png"):
        import matplotlib.pyplot as plt
        l = len(images)
        w = 6
        h = max(l//6,1)
        plt.figure(figsize=(20, h*2))
        for i,image in enumerate(images):
            # display original
            ax = plt.subplot(h,w,i+1)
            plt.imshow(image,interpolation='nearest',cmap='gray',)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(name)
    configs = generate_configs(6)
    puzzles = generate(configs, 2, 3)
    print(puzzles[10])
    plot_image(puzzles[10],"digital_puzzle.png")
    plot_grid(puzzles[:36],"digital_puzzles.png")
    _transitions = transitions(2,3)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show,"digital_puzzle_transitions.png")
