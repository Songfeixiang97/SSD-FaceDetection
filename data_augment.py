#!/usr/bin/env python
# coding: utf-8
from imgaug import *
import imgaug.augmenters as iaa

def data_aug(images):
    
    seq = iaa.Sometimes(
    
        0.5,
        iaa.Identity(),
    
        iaa.Sometimes(0.5,
                      iaa.Sequential([
    
                          iaa.Sometimes(0.5,
                                        iaa.OneOf([
    
                                            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
                                            iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255)),
                                            iaa.ReplaceElementwise(0.03, [0, 255]),
                                            iaa.GaussianBlur(sigma=(0.0, 3.0)),
    
                                            iaa.BilateralBlur(
    d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
                                        ])
    
                                       ),
    
                          iaa.OneOf([iaa.Add((-40, 40)),
    
                                     iaa.AddElementwise((-20, 20)),
    
                                     iaa.pillike.EnhanceBrightness()       
                          ]),
    
                          iaa.OneOf([
    
                              iaa.GammaContrast((0.2, 2.0)),
    
                              iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
    
                              iaa.LogContrast(gain=(0.6, 1.4)),
    
                              iaa.AllChannelsCLAHE(),
    
                              iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                              
                          ])
                      ])
                     )
    )
    
    images = seq(images = images)
    
    return images

def data_aug2(image):
    
    seq = iaa.Sometimes(
    
        0.5,
    
        iaa.Identity(),
    
        iaa.OneOf([
    
            iaa.CoarseDropout((0.1, 0.2), size_percent=(0.01, 0.02)),
    
            iaa.CoarseSaltAndPepper(0.1, size_percent=(0.01, 0.02))
    
        ])
    )
    
    image = seq(image = image)
    
    return image
