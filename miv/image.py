import io
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from .reductions import *

class Image2D:
    def __init__(self,image_key,cmap,vmin,vmax,reduce,alpha=1.0,mask_alpha=False,colorbar=None):
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.image_key = image_key
        self.reduce = reduce
        self.alpha = alpha
        self.mask_alpha = mask_alpha
        self.colorbar = colorbar

        assert self.alpha == "mask" or isinstance(self.alpha,float) or self.alpha is None

    def _get_2d_image(self,data):
        if isinstance(self.image_key, list):
            return [self.reduce(data[k],data) for k in self.image_key]
        
        return self.reduce(data[self.image_key], data)
    
    def process_array(self,arr):
        return arr
    
    def legend(self,ax):
        pass

    def plot(self,data, ax):
        arr = self._get_2d_image(data)
        arr = self.process_array(arr)
        alpha = None
        if self.alpha is not None:
            alpha = self.alpha*np.ones_like(arr).astype("float")
        if self.mask_alpha:
            alpha *= (arr>0)
        img = ax.imshow(arr,cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,alpha=alpha)
        self.legend(ax)
        if self.colorbar is not None:
            self.colorbar(img, ax)
        return img

class Colorbar:
    def __init__(self,
                 orientation="horizontal",
                 ticks=3,
                 title=None,
                 width="90%",
                 height=0.15,
                 loc="lower center",
                 borderpad=-5,
                 labelpad=-42,
                 **kwargs):
        
        self.cbar_orientation = orientation
        self.cbar_ticks = ticks
        self.cbar_title = title
        self.legend_args = {"loc":loc, "borderpad":borderpad,"width":width,"height":height}
        self.kwargs = kwargs
        self.labelpad = labelpad

    def __call__(self,im,ax):
        axins = inset_axes(ax,
                    **self.legend_args,
                    **self.kwargs
                   )
    
        cbar = plt.colorbar(im, cax=axins, orientation=self.cbar_orientation)
        self._set_ticks(cbar,im)
        if self.cbar_title is not None:
            cbar.set_label(self.cbar_title,labelpad=self.labelpad)

    def _set_ticks(self,cbar,im):
        if isinstance(self.cbar_ticks, list):
            cbar.set_ticks(self.cbar_ticks)
        elif isinstance(self.cbar_ticks, int):
            cbar.set_ticks(np.linspace(im.norm.vmin,im.norm.vmax,self.cbar_ticks))
        else:
            cbar.set_ticks([im.norm.vmin,im.norm.vmax])
        

