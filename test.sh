python ./tools/test_net.py --gpu 0 --def prototxt/test.prototxt --net output/default/voc_2007_trainval/caffenet_det_adapt_iter_40000.caffemodel 2>&1 | tee log_test.txt
