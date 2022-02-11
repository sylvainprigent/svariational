

import torch
import torch.nn as nn
import torch.optim as optim


def total_variation_loss(img):
    a, b, h, w = img.size()
    tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
    tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
    return (tv_h+tv_w)/(a*b*h*w)


class TotalVariationDenoising:
    def __init__(self, regularization = 0.01) -> None:
        self.regularization = regularization

    def run(self, image):
        denoised_image = image.detach().clone()
        denoised_image.requires_grad = True
        learning_rate = 0.01
        optimizer = optim.Adam([denoised_image], lr=learning_rate)
        loss_function = nn.MSELoss()
        for _ in range(1000):
            optimizer.zero_grad()
            loss = (1-self.regularization)*loss_function(denoised_image, image) + self.regularization * total_variation_loss(denoised_image)
            loss.backward()
            optimizer.step()
        return denoised_image    

    def energy(self, image_in, image_out):
        loss_function = nn.MSELoss()
        loss = (1-self.regularization)*loss_function(image_out, image_in) + self.regularization * total_variation_loss(image_out)
        return loss.detach().numpy()
