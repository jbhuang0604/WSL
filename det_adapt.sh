python ./tools/train_net.py --gpu 0 --solver prototxt/det_adapt_solver.prototxt --weights caffenet_cls_adapt_iter_10000.caffemodel 2>&1 | tee log_det_adapt.txt
