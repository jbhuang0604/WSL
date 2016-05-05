#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxPairWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_pair_param(this->layer_param_);
  softmax_pair_param.set_type("SoftmaxPair");
  softmax_pair_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_pair_param);
  softmax_pair_bottom_vec_.clear();
  softmax_pair_bottom_vec_.push_back(bottom[0]);
  softmax_pair_top_vec_.push_back(&prob_);
  softmax_pair_layer_->SetUp(softmax_pair_bottom_vec_, softmax_pair_top_vec_);
}

template <typename Dtype>
void SoftmaxPairWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_pair_layer_->Reshape(softmax_pair_bottom_vec_, softmax_pair_top_vec_);
  softmax_pair_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_pair_param().axis());
  if (top.size() >= 2) {
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxPairWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_pair_layer_->Forward(softmax_pair_bottom_vec_, softmax_pair_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    loss += -log(std::max(prob_data[i * dim + static_cast<int>(label[i])],
                          Dtype(FLT_MIN)));
  }
  if (top.size() >= 1) {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxPairWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    for (int i = 0; i < num; ++i) {
      int gt = static_cast<int>(label[i]);
      bottom_diff[i * dim + gt] -= 1;
      for (int j = 0; j < dim; ++j) {
        if ((int)(j/2) != (int)(gt/2))
          bottom_diff[i*dim+j] = Dtype(0);
      }
    }
    // Scale down gradient
    caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(SoftmaxPairWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxPairWithLoss);

}  // namespace caffe
