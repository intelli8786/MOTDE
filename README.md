# Enhancement of multi-target tracking performance in dynamic environments
![Method](https://user-images.githubusercontent.com/25438139/88682892-2b194a80-d12e-11ea-9469-57129793df2e.png)
This repository contains the official implementation of the following paper:


> **Enhancement of multi-target tracking performance in dynamic environments**<br>
> Ji Seong Kim, Doo Soo Chang, Yong Suk Choi<br>
> 
> **Abstract:** *In this paper, we propose several methods to improve the performance of Multiple Object Tracking (MOT) in dynamic environments such as robots and autonomous vehicles. The first method is to restore and re-detect detection results that are difficult to reliably trust to improve the detection. The second is to restore noisy regions in the image before the tracking association to improve the identification. To implement the image restoration function used in these two methods, an image inference model based on SRGAN(Super Resolution Generative Adversarial Networks) is used. Finally, the third method includes an association method using face features to reduce failures in the tracking association. Three distance metrics are designed so that this method can be applied to various environments. In order to validate the effectiveness of our proposed methods, we select two baseline trackers for comparative experiments and construct a robotic environment that interacts with real people and provides services. Experimental results demonstrate that the proposed methods efficiently overcome dynamic situations and show favorable performance in general situations.*

## Usage
### Development Environment
* CPU : i7 8086k
* GPU : NVIDIA GeForce RTX2080ti
* Ubuntu 18.04
### Dependency
* Cuda 9.0
* CuDNN 7.0.5
* python 3.6.4
* tensorflow-gpu 1.9.0
* tensorlayer 1.9.0
* numpy 1.16.1
* opencv 4.2.0
* llvmlite 0.26.0
* numba 0.41.0
* h5py 2.8.0
* scipy 1.1.0
* cffi 1.14.1
* motmetrics 1.2.0
* torch 0.4.1
* torchvision 0.2.1
* darkflow 1.0.0

### Setup

```bash
$ apt install cuda-cublas-dev-9-0
$ apt-get install cuda-cusparse-dev-9-0
$ cd based_MOTDT
$ python3 bbox_setup.py build_ext --inplace
$ sh make.sh
```

### Evaluation
* Save the pre-trained weight file in the **Weights** directory. [[Download](https://1drv.ms/u/s!Av_TIPQTjQYp-Kp1xNug5hcoVyd7Dw?e=liJv8d)]
* Save the robot environment benchmark set in the **Benchmark_Set** directory. [[Download](https://1drv.ms/u/s!Av_TIPQTjQYp-Kp3eLpY-rmQ6cucJg?e=TJmRgf)]

##### Evaluation Example:
```bash
$ python3 based_MOTDT/Evaluation.py
$ python3 based_DeepSORT/Evaluation.py
```
* Benchmark results are stored in the **Benchmark_Results** directory.
* We provide all the benchmark results obtained through experimentation. [[Download](https://1drv.ms/u/s!Av_TIPQTjQYp-Kp2_SMD-wzEpXb-hA?e=QdHzvF)]


### Train
* Save the our **image restoration training dataset** in the **Image_Restoration_Module** directory. [[Download](https://1drv.ms/u/s!Av_TIPQTjQYp-Kp4WmomrtYeh1sIRg?e=srVA5r)]
* If you want to check the inference performance of the model during training, save the test image in the **Image_Restoration_Module/inference_input** directory in advance.
* Inference results are stored during training in the **Image_Restoration_Module/inference_output** directory.
##### Image Restoration Module Traning Example:
```bash
$ python3 Image_Restoration_Module/Train.py
```
* When you're done, move the weights file to the **Weights** directory.


## Results
### Quantitative Results

![image](https://user-images.githubusercontent.com/25438139/88683010-4d12cd00-d12e-11ea-89d5-935b7d70801c.png)

### Qualitative Results

![Experiment_QualitativeResults](https://user-images.githubusercontent.com/25438139/88683044-56039e80-d12e-11ea-994e-e83cbd020542.png)