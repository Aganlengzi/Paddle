// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/ascend_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& dim,
                    bool keep_dim,
                    phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MeanRawGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& out_grad,
                       const std::vector<int64_t>& dim,
                       bool keep_dim,
                       bool reduce_all,
                       phi::DenseTensor* x_grad) {}

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::ScalarArray& axes_array,
                 const phi::ScalarArray& starts_array,
                 const phi::ScalarArray& ends_array,
                 phi::DenseTensor* out) {}

template <typename T, typename Context>
void SGDKernel(const Context& dev_ctx,
               const phi::DenseTensor& param_var,
               const phi::DenseTensor& learning_rate,
               const phi::DenseTensor& grad_var,
               phi::DenseTensor* param_out) {}

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {}

#define DECALRE_COMPARE_KERNEL(compare_kernel) \
  template <typename T, typename Context>      \
  void compare_kernel(const Context& ctx,      \
                      const DenseTensor& x,    \
                      const DenseTensor& y,    \
                      int axis,                \
                      DenseTensor* out) {}

DECALRE_COMPARE_KERNEL(LessThanKernel)
DECALRE_COMPARE_KERNEL(GreaterEqualKernel)
DECALRE_COMPARE_KERNEL(EqualKernel)
DECALRE_COMPARE_KERNEL(NotEqualKernel)
#undef DECALRE_COMPARE_KERNEL

template <typename T, typename Context>
void SoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss,
                                   phi::DenseTensor* backprop) {}

template <typename T, typename Context>
void SoftmaxWithCrossEntropyGradKernel(const Context& dev_ctx,
                                       const phi::DenseTensor& labels,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& backprop,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {}

}  // namespace phi

PD_REGISTER_KERNEL(
    max_raw, CPU, ALL_LAYOUT, phi::MaxRawKernel, float, double, bool) {}

PD_REGISTER_KERNEL(
    min_raw, CPU, ALL_LAYOUT, phi::MinRawKernel, float, double, bool) {}

PD_REGISTER_KERNEL(max, CPU, ALL_LAYOUT, phi::MaxKernel, float, double, bool) {}

PD_REGISTER_KERNEL(min, CPU, ALL_LAYOUT, phi::MinKernel, float, double, bool) {}

PD_REGISTER_KERNEL(mean_raw_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MeanRawGradKernel,
                   float,
                   double,
                   bool) {}

PD_REGISTER_KERNEL(
    mean_grad, CPU, ALL_LAYOUT, phi::MeanGradKernel, float, double, bool) {}

PD_REGISTER_KERNEL(
    slice, CPU, ALL_LAYOUT, phi::SliceKernel, float, double, bool) {}

PD_REGISTER_KERNEL(sgd, CPU, ALL_LAYOUT, phi::SGDKernel, float, double, bool) {}

PD_REGISTER_KERNEL(
    argsort, CPU, ALL_LAYOUT, phi::ArgsortKernel, float, double) {}

PD_REGISTER_KERNEL(less_than,
                   CPU,
                   ALL_LAYOUT,
                   phi::LessThanKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(greater_equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::GreaterEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::EqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
PD_REGISTER_KERNEL(not_equal,
                   CPU,
                   ALL_LAYOUT,
                   phi::NotEqualKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}

PD_REGISTER_KERNEL(softmax_with_cross_entropy,
                   CPU,
                   ALL_LAYOUT,
                   phi::SoftmaxWithCrossEntropyKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(softmax_with_cross_entropy_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SoftmaxWithCrossEntropyGradKernel,
                   float) {}
