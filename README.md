# The Contextual Loss [[project page]](http://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Contextual/)

This is a Tensorflow implementation of the Contextual loss function as reported in the following papers:

## The Contextual Loss for Image Transformation with Non-Aligned Data, [arXiv](https://arxiv.org/abs/1803.02077)
## Learning to Maintain Natural Image Statistics, [arXiv](https://arxiv.org/abs/1803.04626)

[Roey Mechrez*](http://cgm.technion.ac.il/people/Roey/), Itamar Talmi*, Firas Shama, [Lihi Zelnik-Manor](http://lihi.eew.technion.ac.il/). [The Technion](http://cgm.technion.ac.il/)

Copyright 2018 Itamar Talmi and Roey Mechrez Licensed for noncommercial research use only.

<div align='center'>
  <img src='teaser.png' height="500px">
</div>

## Setup

### Background
This code is mainly the contextual loss function. The two papers have many applications, here we provide only one applications: animation from single image.

An example pre-trained model can be download from this [link](https://www.dropbox.com/s/37nz4hy7ai4pqxc/single_im_D32_42_1.0_DC42_1.0.zip?dl=0)

The data for this example can be download from this [link](https://www.dropbox.com/s/ggb6v6rv1a0212y/single.zip?dl=0)

### Requirement
Required python libraries: Tensorflow (>=1.0) + Scipy + Numpy
os, math, json, re, easydict, logging, enum are also needed

Tested in Windows + Intel i7 CPU + Nvidia Titan Xp (and 1080ti) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.


### Quick Start (Testing)
1. Clone this repository.
2. Download the pretrained model from this [link](https://www.dropbox.com/s/q3wjtaxr76cdx3t/imagenet-vgg-verydeep-19.mat?dl=0)
3. Extract the zip file under ```result``` folder. The models should be in ```based_dir/result/single_im_D32_42_1.0_DC42_1.0/```
3. Update the ```config.base_dir``` and ```config.vgg_model_path``` in ```config.py``` and run: ``` demo_256_single_image.py```

### Training
1. Change ```config.TRAIN.to_train``` to ```True```
2. Arrange the paths to the data, should have ```train``` and ```test``` folders
2. run ``` demo_256_single_image.py ``` for 10 epochs. 


## License

   This software is provided under the provisions of the Lesser GNU Public License (LGPL). 
   see: http://www.gnu.org/copyleft/lesser.html.

   This software can be used only for research purposes, you should cite
   the aforementioned papers in any resulting publication.

   The Software is provided "as is", without warranty of any kind.

   
## Citation
If you use our code for research, please cite our paper:
```
@article{mechrez2018contextual,
  title={The Contextual Loss for Image Transformation with Non-Aligned Data},
  author={Mechrez, Roey and Talmi, Itamar and Zelnik-Manor, Lihi},
  journal={arXiv preprint arXiv:1803.02077},
  year={2018}
}
@article{mechrez2018Learning,
  title={Learning to Maintain Natural Image Statistics, [arXiv](https://arxiv.org/abs/1803.04626)},
  author={Mechrez, Roey and Talmi, Itamar and Shama, Firas and Zelnik-Manor, Lihi},
  journal={arXiv preprint arXiv:1803.04626},
  year={2018}
}
```

   
## Code References

[1] Template Matching with Deformable Diversity Similarity, https://github.com/roimehrez/DDIS

[2] Photographic Image Synthesis with Cascaded Refinement Networks https://cqf.io/ImageSynthesis/

