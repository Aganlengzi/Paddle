/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/api/ext/kernel_meta_info.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/custom_kernel.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

////////////////////// Kernel Meta Info //////////////////////

KernelMetaInfo& KernelMetaInfo::SetKernelFn(CustomKernelFunc&& func) {
  kernel_fn_ = std::forward<CustomKernelFunc>(func);
  return *this;
}

//////////////// Kernel Meta Info Map /////////////////

std::vector<KernelMetaInfo>& KernelMetaInfoMap::operator[](
    const std::string& name) {
  return map_[name];
}

const std::unordered_map<std::string, std::vector<KernelMetaInfo>>&
KernelMetaInfoMap::GetMap() const {
  return map_;
}

//////////////// Kernel Meta Info Builder /////////////////

KernelMetaInfoBuilder::KernelMetaInfoBuilder(std::string&& op_name,
                                             const std::string& backend,
                                             const std::string& data_layout,
                                             const std::string& data_type) {
  // 1. member assign
  if (backend == "CPU") {
    backend_ = pten::Backend::CPU;
  }
  if (backend == "NPU") {
    backend_ = pten::Backend::NPU;
  }
  if (data_layout == "ANY") {
    layout_ = pten::DataLayout::ANY;
  }
  if (data_type == "float") {
    dtype_ = pten::DataType::FLOAT32;
  }
  if (data_type == "int64_t") {
    dtype_ = pten::DataType::INT64;
  }
  op_name_ = std::forward<std::string>(op_name);

  // 2. check and meta info build
  auto& info_vector = KernelMetaInfoMap::Instance()[op_name_];
  auto kernel_meta = KernelMetaInfo(op_name_, backend_, layout_, dtype_);
  info_vector.emplace_back(std::move(kernel_meta));
  // 3. get current info ptr
  info_ptr_ = &(info_vector.back());
}

KernelMetaInfoBuilder::KernelMetaInfoBuilder(std::string&& op_name,
                                             pten::Backend backend,
                                             pten::DataLayout data_layout,
                                             pten::DataType data_type) {
  // 1. member assign
  op_name_ = std::forward<std::string>(op_name);
  backend_ = backend;
  layout_ = data_layout;
  dtype_ = data_type;

  // 2. check and meta info build
  auto& info_vector = KernelMetaInfoMap::Instance()[op_name_];
  auto kernel_meta = KernelMetaInfo(op_name_, backend_, layout_, dtype_);
  info_vector.emplace_back(std::move(kernel_meta));
  // 3. get current info ptr
  info_ptr_ = &(info_vector.back());
}

KernelMetaInfoBuilder& KernelMetaInfoBuilder::SetKernelFn(
    CustomKernelFunc func) {
  info_ptr_->SetKernelFn(std::forward<CustomKernelFunc>(func));
  return *this;
}

/////////////////////// Kernel register API /////////////////////////

void RegisterAllCustomKernel() {
  auto& kernel_meta_info_map = KernelMetaInfoMap::Instance();
  framework::RegisterKernelWithMetaInfoMap(kernel_meta_info_map);
}

void LoadCustomKernelLib(const std::string& dso_name) {
  paddle::framework::LoadKernelMetaInfoAndRegisterKernel(dso_name);
}
}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

// PD_Status
PD_Status* PD_NewStatus() { return new PD_Status; }

void PD_DeleteStatus(PD_Status* s) { delete s; }

// PD_Tensor
int64_t PD_TensorElementCount(const PD_Tensor* t) { return t->tensor->numel(); }

std::vector<int64_t> PD_TensorShape(const PD_Tensor* t) {
  return t->tensor->shape();
}

void* PD_TensorData(const PD_Tensor* t) {
  return std::dynamic_pointer_cast<pten::DenseTensor>(t->tensor->impl())
      ->mutable_data();
}

// PD_Register
int PD_NumInputs(const PD_ExecutionContext* ctx) {
  auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
      const_cast<PD_ExecutionContext*>(ctx));
  auto innamelist = cc_ctx->InNameList();
  for (auto& input : innamelist) {
    std::cout << "PD_NumInputs: " << input << std::endl;
  }
  return static_cast<int>(innamelist.size());
}

void PD_GetInput(const PD_ExecutionContext* ctx,
                 const std::string& name,
                 PD_Tensor** tensor) {
  auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
      const_cast<PD_ExecutionContext*>(ctx));

  auto* x = cc_ctx->Input<paddle::framework::Tensor>(name);

  PADDLE_ENFORCE_NOT_NULL(x,
                          paddle::platform::errors::NotFound(
                              "Input tensor (%s) is nullptr.", name));
  PADDLE_ENFORCE_EQ(x->IsInitialized(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "Input tensor (%s) is not initialized.", name));

  auto* ret = new paddle::Tensor();
  ret->set_impl(std::move(paddle::experimental::MakePtenDenseTensor(*x)));
  *tensor = new PD_Tensor{ret};
}

void* PD_GetStream(const PD_ExecutionContext* ctx) {
  auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
      const_cast<PD_ExecutionContext*>(ctx));

#ifdef PADDLE_WITH_ASCEND_CL
  return cc_ctx->template device_context<paddle::platform::NPUDeviceContext>()
      .stream();
#endif
}

void PD_SetOutput(const PD_ExecutionContext* ctx,
                  const std::string& name,
                  paddle::Tensor& out) {
  auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
      const_cast<PD_ExecutionContext*>(ctx));

  auto* x = cc_ctx->Output<paddle::framework::Tensor>(name);
  PADDLE_ENFORCE_NOT_NULL(x,
                          paddle::platform::errors::NotFound(
                              "Output tensor (%s) is nullptr.", name));

  paddle::experimental::MovesStorage(
      std::dynamic_pointer_cast<pten::DenseTensor>(out.impl()).get(), x);
}

std::vector<int64_t> PD_GetOutputShape(const PD_ExecutionContext* ctx,
                                       const std::string& name) {
  auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
      const_cast<PD_ExecutionContext*>(ctx));
  auto* x = cc_ctx->Output<paddle::framework::Tensor>(name);
  PADDLE_ENFORCE_NOT_NULL(x,
                          paddle::platform::errors::NotFound(
                              "Output tensor (%s) is nullptr.", name));
  return paddle::framework::vectorize<int64_t>(x->dims());
}

PD_Tensor* PD_AllocateOutput(const PD_ExecutionContext* ctx,
                             const std::string& name,
                             PD_DataType dtype,
                             const int64_t* dims,
                             int num_dims) {
  auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
      const_cast<PD_ExecutionContext*>(ctx));
  auto* x = cc_ctx->Output<paddle::framework::Tensor>(name);
  PADDLE_ENFORCE_NOT_NULL(x,
                          paddle::platform::errors::NotFound(
                              "Output tensor (%s) is nullptr.", name));
  if (num_dims) {
    std::vector<int64_t> shape(dims, dims + num_dims);
    x->Resize(paddle::framework::make_ddim(shape));
  }
  x->mutable_data(
      cc_ctx->GetPlace(),
      pten::TransToProtoVarType(static_cast<pten::DataType>(dtype)));
  auto* ret = new paddle::Tensor();
  ret->set_impl(std::move(paddle::experimental::MakePtenDenseTensor(*x)));
  return new PD_Tensor{ret};
}

#ifndef _WIN32
// C-API to get global KernelMetaInfoMap.
paddle::KernelMetaInfoMap& PD_GetKernelMetaInfoMap() {
  return paddle::KernelMetaInfoMap::Instance();
}
#endif

#ifdef __cplusplus
}  // end extern "C"
#endif
