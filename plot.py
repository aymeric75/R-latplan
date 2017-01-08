import numpy as np

def fix_images(images,dims=None):
    if isinstance(images,list) or isinstance(images,tuple):
        expanded = []
        for i in images:
            expanded.extend(fix_image(i,dims))
        return expanded
    if len(images.shape) == 3:
        return images
    if len(images.shape) == 4:
        return np.einsum("bxyc->bcxy",images).reshape((-1,)+images.shape[1:3])
    if len(images.shape) == 2:
        return images.reshape((images.shape[0],)+dims)
    raise BaseException("images.shape={}, dims={}".format(images.shape,dims))

def fix_image(image,dims=None):
    if len(image.shape) == 2:
        return np.expand_dims(image,axis=0)
    if len(image.shape) == 3:
        return np.einsum("xyc->cxy",image).reshape((-1,)+image.shape[0:2])
    if len(image.shape) == 1:
        return image.reshape((1,)+dims)
    raise BaseException("image.shape={}, dims={}".format(image.shape,dims))

def plot_grid(images,w=10,path="plan.png"):
    import matplotlib.pyplot as plt
    l = 0
    images = fix_images(images)
    l = len(images)
    h = l//w+1
    plt.figure(figsize=(w, h))
    for i,image in enumerate(images):
        ax = plt.subplot(h,w,i+1)
        try:
            plt.imshow(image,interpolation='nearest',cmap='gray',)
        except TypeError:
            TypeError("Invalid dimensions for image data: image={}".format(np.array(image).shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(path)

def plot_grid2(images,w=10,shape=None,path="plan.png"):
    import matplotlib.pyplot as plt
    images = fix_images(images,shape)
    l = images.shape[0]
    h = l//w
    margin = 3
    m_shape = (margin + np.array(shape))
    figure = np.ones(m_shape * np.array((h,w)))
    # print images.shape,h,w
    for y in range(h):
        for x in range(w):
            begin = m_shape * np.array((y,x))
            end   = (m_shape * (np.array((y,x))+1)) - margin
            # print begin,end,y*w+x
            figure[begin[0]:end[0],begin[1]:end[1]] = images[y*w+x]
    plt.figure(figsize=(h,w))
    plt.imshow(figure,interpolation='nearest',cmap='gray',)
    plt.savefig(path)