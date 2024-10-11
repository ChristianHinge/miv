from .image import *
import matplotlib.patches as mpatches

class CT(Image2D):
    def __init__(self,image_key, reduce, preset="soft",colorbar=None):
        if preset == "soft":
            hu_min = -250
            hu_max = 350
        super().__init__(image_key,"gray",hu_min,hu_max,reduce,colorbar=colorbar)

class PET(Image2D):
    def __init__(self,image_key, suv_max,reduce,overlay=False,colorbar=None):
        if overlay:
            alpha = 0.5
            cmap = "hot"
        else:
            alpha = 1.0
            cmap = "gray_r"
        super().__init__(image_key,cmap,0,suv_max,reduce,alpha,colorbar=colorbar)

class Segmentation(Image2D):
    def __init__(self,image_key, reduce, cmap, alpha=0.5,colorbar=None):
        super().__init__(image_key,cmap,0,1,reduce,mask_alpha=True,alpha=alpha,colorbar=colorbar)

class SegmentationPrediction(Image2D):
    def __init__(self,true_seg_key,pred_seg_key,reduce,legend=True,alpha=0.5,legend_args=None,colorbar=None):
        if legend_args is None:
            legend_args = {"loc":"lower center", "borderaxespad":0., "ncol":3, "bbox_to_anchor":(0.5, -0.1)}
        self.legend_args = legend_args
        self._legend=legend
        self._alpha=alpha
        super().__init__([true_seg_key,pred_seg_key],None,None,None,reduce,alpha=None,mask_alpha=False,colorbar=colorbar)

    def process_array(self,arr):
        y_true, y_pred = arr
        TP = (y_true == 1) & (y_pred == 1)
        FP = (y_true == 0) & (y_pred == 1)
        FN = (y_true == 1) & (y_pred == 0)
        result_image = np.zeros((y_true.shape[0], y_true.shape[1], 4))  # RGB image

        # Assign colors: TP = Green, FP = Red, FN = Blue
        result_image[TP] = [0, 1, 0, self._alpha]  # Green
        result_image[FN] = [1, 0, 0, self._alpha]  # Red
        result_image[FP] = [0, 0.6, 1, self._alpha]  # Blue
        return result_image
    
    def legend(self,ax):
        if self._legend:
            colors = [ [0,1,0],[1,0,0],[0,0.6,1]]
            values = ["TP","FN","FP"]
            patches = [ mpatches.Patch(color=colors[i],label=values[i]) for i in range(len(values)) ]
            ax.legend(handles=patches, **self.legend_args)


class Contour(Image2D):
    def __init__(self,image_key, reduce, cmap, alpha=0.5,width=3,dilate=0):
        self.width = width
        self.dilate = dilate
        super().__init__(image_key,cmap,0,1,reduce,mask_alpha=True,alpha=alpha,colorbar=None)

    def process_array(self,arr):
        im = np.ascontiguousarray(np.uint8((arr>0).astype(float)*255))
        #dilate im
        kernel = np.ones((3,3),np.uint8)
        im = cv.dilate(im,kernel,iterations = self.dilate)
        ret, thresh = cv.threshold(im, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cont = np.zeros_like(im)
        cv.drawContours(cont, contours, -1, (255,255,0), self.width)
        return cont/255


