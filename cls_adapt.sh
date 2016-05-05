./caffe-wsl/build/tools/caffe train -gpu 0 -solver prototxt/cls_adapt_solver.prototxt -weights data/imagenet_models/bvlc_reference_caffenet.caffemodel 2>&1 | tee log_cls_adapt.txt
