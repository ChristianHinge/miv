import torch
import math
import torchio as tio 
from matplotlib import pyplot as plt
import io
import numpy as np

def plot(windows, data, show=True,save=None, nrows=1,return_array=False,dpi=400,figsize=(10,10)):
    imgs_per_row = math.ceil(len(windows)/nrows)
    figure, plt_axes = plt.subplots(nrows,imgs_per_row,figsize=figsize, dpi=dpi)

    for k,v in data.items():
        if isinstance(v, tio.Image):
            data[k] = v["data"].numpy()
        if isinstance(v, torch.Tensor):
            data[k] = v.numpy()
        data[k] = np.squeeze(data[k])

    for plt_ax, ax in zip(plt_axes.flatten(),windows):
        ax.set_ax(plt_ax)
        ax.plot(data)
        plt_ax.axis("off")
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save,dpi=dpi)

    if return_array:
        io_buf = io.BytesIO()
        figure.savefig(io_buf, format='raw', dpi=dpi)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
        io_buf.close()
        return img_arr
    
def remove_white_space(img,horizontal_padding=6,vertical_padding=6):
    ix = (img==255).all(axis=(0,2))
    ix[horizontal_padding:] &= ix[:-horizontal_padding]
    ix[:-horizontal_padding] &= ix[horizontal_padding:]
    img = img[:,~ix,:]

    ix = (img==255).all(axis=(1,2))
    ix[vertical_padding:] &= ix[:-vertical_padding]
    ix[:-vertical_padding] &= ix[vertical_padding:]
    img = img[~ix,:,:]
    return img
