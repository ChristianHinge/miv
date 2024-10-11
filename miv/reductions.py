import numpy as np

class Dim:
    CORONAL = 1
    SAGITTAL = 0
    AXIAL = 2

# abstract class
class ReduceOperation:
    def __init__(self, axis,xlim=None,ylim=None):
        self.axis=axis
        self.xlim = xlim
        self.ylim = ylim

    def _maybe_crop(self, img):
        if self.xlim is not None:
            img = img[:,self.xlim[0]:self.xlim[1]]
        if self.ylim is not None:
            img = img[self.ylim[0]:self.ylim[1]]
        return img
        
    def __call__(self, img, data):
        img = self.reduce(img, data)
        img = np.rot90(img)
        img = self._maybe_crop(img)
        return img 
    
    def reduce(self, img, data):
        raise NotImplementedError

class ArgX(ReduceOperation):
    def __init__(self, axis, ref_key, reduce_op,xlim=None,ylim=None):
        super().__init__(axis,xlim,ylim)
        self.ref_key = ref_key
        self.reduce_op = reduce_op

    def reduce(self, img, data):
        ref = data[self.ref_key]
        ix = self.reduce_op(ref, axis=self.axis)
        return np.take(img,ix,self.axis)

class ArgMin(ArgX):
    def __init__(self, axis, ref_key,xlim=None,ylim=None):
        super().__init__(axis, ref_key, np.argmin,xlim,ylim)

class ArgMax(ArgX):
    def __init__(self, axis, ref_key,xlim=None,ylim=None):
        super().__init__(axis, ref_key, np.argmax,xlim,ylim)

class Slice(ReduceOperation):
    def __init__(self, axis, slice_ix,xlim=None,ylim=None):
        super().__init__(axis,xlim,ylim)
        self.slice_ix = slice_ix

    def reduce(self, img, data):
        return np.take(img,self.slice_ix,self.axis)
    
class Max(ReduceOperation):
    def __init__(self, axis,xlim=None,ylim=None):
        super().__init__(axis,xlim,ylim)

    def reduce(self, img, data):
        return np.max(img,axis=self.axis)

class Min(ReduceOperation):
    def __init__(self, axis,xlim=None,ylim=None):
        super().__init__(axis,xlim,ylim)

    def reduce(self, img, data):
        return np.min(img,axis=self.axis)
    
class Mean(ReduceOperation):
    def __init__(self, axis,xlim=None,ylim=None):
        super().__init__(axis,xlim,ylim)

    def reduce(self, img, data):
        return np.mean(img,axis=self.axis)

class Center(ReduceOperation):
    def __init__(self, axis,xlim=None,ylim=None):
        super().__init__(axis,xlim,ylim)

    def reduce(self, img, data):
        ix = img.shape[self.axis]//2
        return np.take(img,ix,self.axis)