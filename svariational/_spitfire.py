
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.restoration import wiener
from ._padding import mirror_hanning_padding


def hv_loss(img, weighting):
    """Sparse Hessian regularization term

    Parameters
    ----------
    img: Tensor
        Tensor of shape BCHX containing the estimated image
    weighting: float
        Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse

    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:,:,2:,1:-1] + 2*img[:,:,1:-1,1:-1] - img[:,:,:-2,1:-1])
    dyy2 = torch.square(-img[:,:,1:-1,2:] + 2*img[:,:,1:-1,1:-1] - img[:,:,1:-1,:-2])
    dxy2 = torch.square(img[:,:,2:,2:] - img[:,:,2:,1:-1] - img[:,:,1:-1,2:] + img[:,:,1:-1,1:-1])
    hv = torch.sqrt(weighting*weighting*(dxx2 + dyy2 + 2*dxy2) + (1-weighting)*(1-weighting)*torch.square(img[:,:,1:-1,1:-1])).sum()
    return hv/(a*b*h*w)


def dataterm_deconv(blurry_image, deblurred_image, psf):
    """Deconvolution L2 dataterm

    Compute the L2 error between the original image and the convoluted reconstructed image

    Parameters
    ----------
    blurry_image: Tensor
        Tensor of shape BCHX containing the original blurry image
    deblurred_image: Tensor
        Tensor of shape BCHX containing the estimated deblured image
    psf: Tensor
        Tensor containing the point spread function

    """
    conv_op = nn.Conv2d(1, 1, kernel_size=psf.shape[2],
                              stride=1,
                              padding=int((psf.shape[2] - 1) / 2),
                              bias=False)
    with torch.no_grad():
        conv_op.weight = nn.Parameter(psf)
    mse = nn.MSELoss()
    return mse(blurry_image, conv_op(deblurred_image))    


class Spitfire2DDenoising:
    """Apply a 2D spitfire denoising 

    The Spitfire optimization is performed with the Adam algorithm

    Parameters
    ----------
    balance: float
        balance between the dataterm and the regularization term. Value is in [0, 1].
        0 no data term, 1 no regularization term
    weighting: float
        Sparsity model weight. Value is in [0, 1].
        0 sparsity term only, 1 Hessian term only   

    Attributs
    ---------
    device: str
        "cuda" or "cpu" device to run the calculation. Default uses the auto-detect of Torch
    max_iter_: int
        Internal parameter to limit the maximum number of iterations of the gradient descente 
        (default: 20000)
    gradient_step_: float
        Value of one step of the gradient descente. (default: 0.01)    
    loss_: float
        Value of the loss (also called energy) function at the end of the calculation
    niter_: int
        Number of iterations at the end of the calculation
         
    """
    def __init__(self, balance = 0.9, weighting = 0.1) -> None:
        self.balance = balance
        self.weighting = weighting
        self.max_iter_ = 20000
        self.gradient_step_ = 0.01
        self.loss_ = None
        self.niter_ = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, image):
        """Apply Spitfire in one Numpy image

        Parameters
        ----------
        image: Tensor
            Torch tensor containing the image to process

        Returns
        -------
        A Torch tensor containing the deconvolved image

        """
        image_torch = torch.from_numpy(image).float()
        image_torch = image_torch.to(self.device).unsqueeze(0).unsqueeze(0)
        out_image_torch =  self.run(image)
        return out_image_torch[0, 0, :, :].detach().numpy()

    def run(self, image):
        """Apply Spitfire in one Torch image

        Parameters
        ----------
        image: Tensor
            Torch tensor containing the image to process

        Returns
        -------
        A Torch tensor containing the deconvolved image

        """
        denoised_image = image.detach().clone()
        denoised_image.requires_grad = True
        optimizer = optim.Adam([denoised_image], lr=self.gradient_step_)
        loss_function = nn.MSELoss()
        previous_loss = sys.float_info.max 
        count_eq = 0
        self.niter_ = 0
        for _ in range(self.max_iter_):
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.balance * loss_function(denoised_image, image) + (1-self.balance) * hv_loss(denoised_image, self.weigthing)
            if abs(loss - previous_loss) < 1e-6:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break    
            loss.backward()
            optimizer.step()
        self.loss_ = loss    
        return denoised_image   


class Spitfire2DDeconv:
    """Apply a 2D spitfire deconvolution 

    The Spitfire optimization is performed with the Adam algorithm

    Parameters
    ----------
    balance: float
        balance between the dataterm and the regularization term. Value is in [0, 1].
        0 no data term, 1 no regularization term
    weighting: float
        Sparsity model weight. Value is in [0, 1].
        0 sparsity term only, 1 Hessian term only  
    padding: tuple
        Size of the mirror-hanning padding used to avoid borders artifacts. None by default. 
        You should use a padding of at least half the size of the deconvolution PSF in each dimension.
        For exemple set padding=(7, 7) for a (15, 15) sized PSF     

    Attributs
    ---------
    device: str
        "cuda" or "cpu" device to run the calculation. Default uses the auto-detect of Torch
    max_iter_: int
        Internal parameter to limit the maximum number of iterations of the gradient descente 
        (default: 20000)
    gradient_step_: float
        Value of one step of the gradient descente. (default: 0.01)    
    loss_: float
        Value of the loss (also called energy) function at the end of the calculation
    niter_: int
        Number of iterations at the end of the calculation
         
    """
    def __init__(self, balance = 0.9, weigthing = 0.1, padding=None) -> None:
        self.balance = balance
        self.weigthing = weigthing
        self.padding = padding
        self.max_iter_ = 20000
        self.gradient_step_ = 0.01
        self.loss_ = None
        self.niter_ = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, psf, image):
        """Calculate the deconvolution on a Numpy image

        Parameters
        ----------
        psf: ndarray
            Point Spread Function
        image: ndarray
            Blurry input image    
        
        """
        # normalize
        mini = np.min(image)
        maxi = np.max(image)
        image_norm = (image - mini)/ (maxi - mini)
        psf_norm = psf / np.max(psf)

        # add padding
        if self.padding is not None:
            image_norm = mirror_hanning_padding(image_norm, self.padding[0], self.padding[1])

        # calculate the initial guess
        initial_guess = wiener(image_norm, psf_norm, balance=0.5, clip=True)
        self._initial_guess = torch.from_numpy(initial_guess).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # convert to torch
        image_torch = torch.from_numpy( image_norm ).float().unsqueeze(0).unsqueeze(0).to(self.device)
        psf_torch = torch.from_numpy(psf_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # run the deconvolution
        image_deconv_torch = self.run(psf_torch, image_torch)

        # back to numpy
        image_deconv_numpy = image_deconv_torch[0, 0, :, :].detach().numpy()

        # remove padding
        if self.padding is not None:
            image_deconv_numpy = image_deconv_numpy[self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        return image_deconv_numpy*(maxi - mini) + mini

    def run(self, psf, image):
        if self._initial_guess is not None:
            deconv_image = self._initial_guess
        else:    
            deconv_image = image.detach().clone()
        deconv_image.requires_grad = True
        optimizer = optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12 
        count_eq = 0
        self.niter_ = 0
        reg1 = self.balance
        #reg2 = math.sqrt(1-self.balance*self.balance)
        reg2 = 1-self.balance
        for _ in range(self.max_iter_):
            self.niter_ += 1
            optimizer.zero_grad()
            loss = reg1*dataterm_deconv(image, deconv_image, psf) + reg2 * hv_loss(deconv_image, self.weigthing)
            print('iter:', self.niter_, ' loss:', loss)
            if abs(loss - previous_loss) < 1e-6:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break    
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss    
        return deconv_image  
