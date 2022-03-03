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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature ReduceMaxOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature(
          "max", {"X"}, {"dim", "out_dtype", "keep_dim"}, {"Out"});
    }
    return KernelSignature("max_raw",
                           {"X"},
                           {"dim", "keep_dim", "reduce_all", "out_dtype"},
                           {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceMinOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature(
          "min", {"X"}, {"dim", "out_dtype", "keep_dim"}, {"Out"});
    }
    return KernelSignature("min_raw",
                           {"X"},
                           {"dim", "keep_dim", "reduce_all", "out_dtype"},
                           {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceMeanGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature("mean_grad",
                             {"X", GradVarName("Out")},
                             {"dim", "keep_dim"},
                             {GradVarName("X")});
    }
    return KernelSignature("mean_raw_grad",
                           {"X", GradVarName("Out")},
                           {"dim", "keep_dim", "reduce_all"},
                           {GradVarName("X")});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature SliceOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "slice", {"Input"}, {"axes", "starts", "ends"}, {"Out"});
}

KernelSignature SGDOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "sgd", {"Param", "LearningRate", "Grad"}, {}, {"ParamOut"});
}

KernelSignature LessThanArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("less_than", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature GreaterEqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("greater_equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature EqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature NotEqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("not_equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature SoftmaxWithCrossEntropyOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("softmax_with_cross_entropy",
                         {"Logits", "Label"},
                         {"soft_label",
                          "use_softmax",
                          "numeric_stable_mode",
                          "ignore_index",
                          "axis"},
                         {"Softmax", "Loss", "Backprop"});
}

KernelSignature SoftmaxWithCrossEntropyGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("softmax_with_cross_entropy_grad",
                         {"Label", "Softmax", "Backprop", GradVarName("Loss")},
                         {"soft_label",
                          "use_softmax",
                          "numeric_stable_mode",
                          "ignore_index",
                          "axis"},
                         {GradVarName("Logits")});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(reduce_max, max);
PD_REGISTER_BASE_KERNEL_NAME(reduce_min, min);
PD_REGISTER_BASE_KERNEL_NAME(reduce_mean_grad, mean_grad);

PD_REGISTER_ARG_MAPPING_FN(reduce_max, phi::ReduceMaxOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reduce_min, phi::ReduceMinOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reduce_mean_grad,
                           phi::ReduceMeanGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(slice, phi::SliceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sgd, phi::SGDOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(less_than, phi::LessThanArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(greater_equal, phi::GreaterEqualArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(equal, phi::EqualArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(not_equal, phi::NotEqualArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy,
                           phi::SoftmaxWithCrossEntropyOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy_grad,
                           phi::SoftmaxWithCrossEntropyGradOpArgumentMapping);
