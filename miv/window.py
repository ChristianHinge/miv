class Window:
    def __init__(self,*images,title=None,title_kwargs={}):
        if not isinstance(images, (tuple, list)):
            images = [images]
        self.images = images
        self.title = title
        self.title_kwargs = title_kwargs

    def set_ax(self,ax):  
        self.ax = ax

    def plot(self, image_dict):
        for img in self.images:
            img.plot(image_dict,self.ax)
        if self.title:
            self.ax.set_title(self.title,**self.title_kwargs)
