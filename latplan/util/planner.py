import pickle
import matplotlib.pyplot as plt
import numpy as np

def ensure_directory(directory):
    if directory[-1] == "/":
        return directory
    else:
        return directory+"/"

sae = None
problem_dir = None
network_dir = None
ama_version = None

import os.path
def problem(path):
    return os.path.join(problem_dir,path)

def network(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(network_dir.replace("/","_"), root, ext)

def ama(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(ama_version, root, ext)




def normalize_with_known_min_max(image, mini, maxi):
    if maxi == mini:
        return image - mini
    else:
        return (image - mini)/(maxi - mini), maxi, mini

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image, mini, maxi):
    image = np.array(image)
    image = image / 255.
    image = image.astype(float)
    image = equalize(image)
    image, orig_max, orig_min = normalize_with_known_min_max(image, mini, maxi)
    image = enhance(image)
    return image, orig_max, orig_min

def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)
    else:
        mean = np.array(mean)
        std = np.array(std)
    return (images - mean)/(std+1e-20), mean, std

def deenhance(enhanced_image):
    temp_image = enhanced_image - 0.5
    temp_image = temp_image / 3
    original_image = temp_image + 0.5
    return original_image

def denormalize(normalized_image, original_min, original_max):
    if original_max == original_min:
        return normalized_image + original_min
    else:
        return (normalized_image * (original_max - original_min)) + original_min

def unnormalize_colors(normalized_images, mean, std): 
    return (normalized_images*std)+mean





def init_goal_misc(p, cycle=1, noise=None, image_path=None, is_soko=False):
    # import sys
    import imageio
    from .plot         import plot_grid
    from .np_distances import bce, mae, mse
    from .noise        import gaussian

    def load_image(name, image_path_):

        with open(image_path_+"/"+name+".p", mode="rb") as f:
            loaded_data = pickle.load(f)
        image = loaded_data["image"]
        # print("IMAGE50")
        # print(image[10:20,10:20,0])
        # exit()



        return image

    def autoencode_image(name,image):



        state = sae.encode(np.array([image]))[0].round().astype(int)
        print("np sum state")
        print(np.sum(state))

        image_rec = sae.decode(np.array([state]))[0]
        print(f"{name} (input) min:",image.min(),"max:",image.max(),)
        print(f"{name} (recon) min:",image_rec.min(),"max:",image_rec.max(),)
        # print(f"{name} BCE:",bce(image,image_rec))
        # print(f"{name} MAE:",mae(image,image_rec))
        print(f"{name} MSE:",mse(image,image_rec))
        # print(state)
        # if image_diff(image,image_rec) > image_threshold:
        #     print("Initial state reconstruction failed!")
        #     sys.exit(3)


        if name =="init":
            init_unorm_color = unnormalize_colors(image, sae.parameters["mean"], sae.parameters["std"])
            
            if not is_soko:
                init_dee = deenhance(init_unorm_color)
                init_unorm_color = denormalize(init_dee, sae.parameters["orig_min"], sae.parameters["orig_max"])
            init_denorm = np.clip(init_unorm_color, 0, 1)
            plt.imsave(problem_dir+"/init-from-State.png", init_denorm)
            plt.close()

        elif name =="goal":
            goal_unorm_color = unnormalize_colors(image, sae.parameters["mean"], sae.parameters["std"])
            
            if not is_soko:
                goal_dee = deenhance(goal_unorm_color)
                goal_unorm_color = denormalize(goal_dee, sae.parameters["orig_min"], sae.parameters["orig_max"])
            goal_unorm_color = np.clip(goal_unorm_color, 0, 1)
            plt.imsave(problem_dir+"/goal-from-State.png", goal_unorm_color)
            plt.close()

        return state, image_rec

    def load_and_encode_image(name):
        image0 = load_image(name,image_path)
        if noise is not None:
            print(f"adding gaussian noise N(0,{noise})")
            image = gaussian(image0, noise)
        else:
            image = image0
        images = [image]
        for i in range(cycle):
            state, image = autoencode_image(name,image)
            images.append(image)
        return image0, state, image, images

    init_image, init, init_rec, init_images = load_and_encode_image("init")
    goal_image, goal, goal_rec, goal_images = load_and_encode_image("goal")


    # sae.plot(np.concatenate([init_images,goal_images]),
    #          path=problem(ama(network(f"init_goal_reconstruction.{cycle}.png"))))
    # if p and not np.all(
    #         p.validate_states(
    #             np.squeeze(     # remove the channel dimension in monochrome domains
    #                 sae.render(
    #                     np.stack(
    #                         [init_rec,goal_rec]))))):
    #     print("Init/Goal state reconstruction failed!")
    #     # sys.exit(3)
    #     print("But we continue anyways...")
    print("INITT ")
    print(init)

    print("GOALL ")
    print(goal)
    return init, goal

def setup_planner_utils(_sae, _problem_dir, _network_dir, _ama_version):
    global sae, problem_dir, network_dir, ama_version
    sae, problem_dir, network_dir, ama_version = \
        _sae, _problem_dir, _network_dir, _ama_version
    return


def puzzle_module(sae):
    import importlib
    assert "generator" in sae.parameters
    p = importlib.import_module(sae.parameters["generator"])
    p.setup()
    return p


import subprocess
def echodo(cmd,*args,**kwargs):
    print(cmd,flush=True)
    subprocess.run(cmd,*args,**kwargs)

def echo_out(cmd):
    print(cmd)
    return subprocess.check_output(cmd)

import time
start = time.time()
times = [{"message":"init","wall":0,"elapsed":0}]
def log(message):
    now = time.time()
    wall = now-start
    elap = wall-times[-1]["wall"]
    times.append({"message":message,"wall":wall,"elapsed":elap})
    print("@[{: =10.3f} +{: =10.3f}] {}".format(wall,elap,message))

