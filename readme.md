# Implementation

* CUDA Version 10.2
  - Tensorflow 1.6.0, gast 0.3.3, Python 3.6.0
  - Tensorflow 2.3.0, Tensorflow-addons 0.12.1, Python 3.6.0

* Download the models

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)

  - [Ensemble adversarial trained models](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)

* Then put these models into "models/"

* Run the code (Please note the pathes in code)
```
python VFI-FGSM.py
```

* The output images are in "output/"