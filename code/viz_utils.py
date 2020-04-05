'''
This module contains general use visualization functions
'''
import numpy as np 
from matplotlib import pyplot as plt
from typing import Iterable, Union

def show_imgs(imgs: Iterable, titles: Iterable=None, n_cols: int=4, colsize: int=2, 
              rowsize: int=2, clip: bool=False, show: bool=True) -> Union[None, tuple]:
    '''
    Visualize iterable of 2D np.ndarray objects using plt.imshow. The plots will be a 
    rectangular grid. 
    
    Parameters
    ------------
    imgs: iterable of images. If you want to plot one image you have to do
          show_imgs([image]).
          
    titles: Optional, iterable of subplot titles (as strings)
    
    n_cols: Optional, number of columns in subplot grid    
    
    colsize: size multiplier in figsize=(colsize*n_cols,...)
    
    rowsize: size multiplier in figsize=(...,rowsize*n_rows)
    
    clip: Optional, if True: Clips every image pixel value to be between (0,1).
          Implicitly, images should be normalized. 
    
    Returns
    ---------
    Returns None if show is True, else it returns (fig, axes)
    '''
    n = len(imgs)
    n_cols=n if n < n_cols else n_cols
    n_rows=int(np.ceil(n/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(colsize*n_cols,rowsize*n_rows))
    axes = np.ravel(axes)
    for ax, img in zip(axes, imgs):
        if clip: img = img.clip(0,1)
        ax.imshow(img)
    
    # Set titles
    if titles is not None:
        [ax.set_title(title) for ax, title in zip(axes, titles)]
        
    # Turn off spines on empty plots
    for ax in axes.ravel()[-(n_cols*n_rows-n):]:
        ax.axis('off')
    
    plt.setp(axes, xticks=[], yticks=[])
    fig.tight_layout()
    
    if show:
        plt.show()
    else:
        return fig, axes