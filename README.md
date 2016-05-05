# Weakly Supervised Object Localization with Progressive Domain Adaptation (CVPR 2016)

This is the research code for the paper:

[Dong Li](https://sites.google.com/site/lidonggg930), [Jia-Bin Huang](https://sites.google.com/site/jbhuang0604), [Yali Li](https://www.researchgate.net/profile/Yali_Li3), [Shengjin Wang](http://www.ee.tsinghua.edu.cn/publish/eeen/3784/2010/20101219115601212198627/20101219115601212198627_.html), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/). "Weakly Supervised Object Localization with Progressive Domain Adaptation" In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

[Project page](https://sites.google.com/site/lidonggg930/wsl)

### Citation

If you find the code and pre-trained models useful in your research, please consider citing:

    @inproceedings{Huang-CVPR-2016,
      author  = {Dong, Li and Huang, Jia-Bin and Li, Yali and Wang, Shengjin and Yang, Ming-Hsuan},
      title   = {Weakly Supervised Object Localization with Progressive Domain Adaptation},
      booktitle = {Proceedings of the IEEE  Conference on Computer Vision and Pattern Recognition)},
      year    = {2015},
      volume  = {},
      number  = {},
      pages   = {}  
      }

### System Requirements

- MATLAB (tested with R2014a on 64-bit Linux)
- Caffe

### Installation

1. Download and unzip the project code.

2. Install caffe. We call the root directory of the project code `WSL_ROOT`.

    ```
    cd $WSL_ROOT/caffe-wsl
    # Now follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html
    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config is in place, then simply do:
    make all -j8
    make pycaffe
    make matcaffe
    ```

3. Download the PASCAL VOC 2007 dataset. Extract all the tars into one directory named `VOCdevkit`. It should have this basic structure: 

    ```
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```

Then create symlinks for the dataset:

    ```
    cd $WSL_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```

4. Download the [pre-trained ImageNet model](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) and put it into `$WSL_ROOT/data/imagenet_models`.

5. Download the pre-computed EdgeBox proposals and put them into `$WSL_ROOT/data/edgebox_data`.

6. Install the project.

    ```
    cd $WSL_ROOT
    # Start MATLAB
    matlab
    >> startup
    ```

### Usage

You will need about 150GB of disk space free for the feature cache (which is stored in `$WSL_ROOT/cache` by default. The final adapted model will be stored in `$WSL_ROOT/output/default/voc_2007_trainval`.

1. Classification adaptation.

    ```
    >> prepare_for_cls_adapt
    cd $WSL_ROOT
    sh cls_adapt.sh
    ```

2. Class-specific proposal mining.

    ```
    >> maskout
    ```

3. MIL for confident proposal mining.

    ```
    >> mil
    ```

4. Detection adaptation.

    ```
    >> prepare_for_det_adapt
    cd $WSL_ROOT
    sh det_adapt.sh
    ```

5. Evaluation.

    ```
    cd $WSL_ROOT
    sh test.sh
    ```
