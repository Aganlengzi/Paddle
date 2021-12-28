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

#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/pten/api/ext/dll_decl.h"
#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/api/include/tensor.h"
#include "paddle/utils/any.h"

#include "paddle/pten/api/ext/op_meta_info.h"

/**
 * Kernel Meta Info Related Define.
 *
 * Used to maintain operator core information.
 *
 */
typedef struct PD_ExecutionContext PD_ExecutionContext;

namespace paddle {
namespace framework {
class PADDLE_API KernelMetaInfoHelper;
}  // namespace framework

using Tensor = paddle::Tensor;

using CustomKernelFunc = void (*)(const PD_ExecutionContext*);

////////////////////// Kernel Meta Info //////////////////////

class PADDLE_API KernelMetaInfo {
 public:
  explicit KernelMetaInfo(const std::string& op_name,
                          pten::Backend backend,
                          pten::DataLayout data_layout,
                          pten::DataType data_type)
      : op_name_(op_name),
        backend_(backend),
        layout_(data_layout),
        dtype_(data_type) {}

  // format: CustomKernelFunc
  KernelMetaInfo& SetKernelFn(CustomKernelFunc&& func);

 private:
  friend class framework::KernelMetaInfoHelper;

  // 1. op name
  std::string op_name_;

  // 2. kernel key info
  pten::Backend backend_{pten::Backend::UNDEFINED};
  pten::DataLayout layout_{pten::DataLayout::UNDEFINED};
  pten::DataType dtype_{pten::DataType::UNDEFINED};

  // 3. custom func info
  CustomKernelFunc kernel_fn_{nullptr};
};

//////////////// Kernel Meta Info Map /////////////////

class PADDLE_API KernelMetaInfoMap {
 public:
  // this function's impl should keep in header file.
  // if move to cc file, meta info can not be added
  // into map
  static KernelMetaInfoMap& Instance() {
    static KernelMetaInfoMap g_custom_kernel_meta_info_map;
    return g_custom_kernel_meta_info_map;
  }

  std::vector<KernelMetaInfo>& operator[](const std::string& name);

  const std::unordered_map<std::string, std::vector<KernelMetaInfo>>& GetMap()
      const;

 private:
  KernelMetaInfoMap() = default;
  std::unordered_map<std::string, std::vector<KernelMetaInfo>> map_;

  PD_DISABLE_COPY_AND_ASSIGN(KernelMetaInfoMap);
};

//////////////// Kernel Meta Info Builder /////////////////

class PADDLE_API KernelMetaInfoBuilder {
 public:
  explicit KernelMetaInfoBuilder(std::string&& op_name,
                                 const std::string& backend,
                                 const std::string& data_layout,
                                 const std::string& data_type);
  explicit KernelMetaInfoBuilder(std::string&& op_name,
                                 pten::Backend backend,
                                 pten::DataLayout data_layout,
                                 pten::DataType data_type);
  KernelMetaInfoBuilder& SetKernelFn(CustomKernelFunc func);

 private:
  // op name
  std::string op_name_;

  // kernel key info
  pten::Backend backend_{pten::Backend::UNDEFINED};
  pten::DataLayout layout_{pten::DataLayout::UNDEFINED};
  pten::DataType dtype_{pten::DataType::UNDEFINED};

  // ref current info ptr
  KernelMetaInfo* info_ptr_;
};

/////////////////////// Kernel register API /////////////////////////

// For inference: compile directly with framework
// Call after PD_REGISTER_KERNEL(...)
void RegisterAllCustomKernel();

// Using this api to load compiled custom kernel's dynamic library and
// register Custom Kernel inside
void LoadCustomKernelLib(const std::string& dso_name);

/////////////////////// Kernel register Macro /////////////////////////

// PD_REGISTER_KERNEL(op_name, MY_DEVICE, NCHW, float)
//     .Inputs({"X"}, {"Y"})
//     .Outputs({"Out"})
//     .Attrs({"axis"})
//     .SetKernelFn(PD_KERNEL(CustomKernelFunc<float>));

#define PD_BUILD_KERNEL(op_name, device, layout, dtype)              \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                    \
      __reg_kernel__##op_name##_##device##_##layout##_##dtype,       \
      "PD_RBUILD_KERNEL must be called in global namespace.");       \
  static ::paddle::KernelMetaInfoBuilder                             \
      __kernel_meta_info_##op_name##_##device##_##layout##_##dtype = \
          ::paddle::KernelMetaInfoBuilder(#op_name, #device, #layout, #dtype)

}  // namespace paddle

///////////////////// C API ///////////////////

#ifdef __cplusplus
extern "C" {
#endif

// pten::DataType
typedef enum PD_DataType {
  PD_UNDEFINED = 0,
  PD_BOOL,
  PD_INT8,   // Char
  PD_UINT8,  // BYte
  PD_INT16,
  PD_INT32,
  PD_UINT32,
  PD_INT64,
  PD_UINT64,
  PD_BFLOAT16,
  PD_FLOAT16,
  PD_UINT16,
  PD_FLOAT32,
  PD_FLOAT64,
  PD_COMPLEX64,
  PD_COMPLEX128,
  PD_NUM_DATA_TYPES
} PD_DataType;

// PD_Status
typedef struct PD_Status {
  int64_t s;

  PD_Status() { s = 0; }
} PD_Status;

PADDLE_API PD_Status* PD_NewStatus();

PADDLE_API void PD_DeleteStatus(PD_Status* s);

// PD_Tensor
typedef struct PD_Tensor { paddle::Tensor* tensor; } PD_Tensor;

PADDLE_API int64_t PD_TensorElementCount(const PD_Tensor* t);

PADDLE_API std::vector<int64_t> PD_TensorShape(const PD_Tensor* t);

PADDLE_API void* PD_TensorData(const PD_Tensor* t);

// PD_Register
// typedef struct PD_ExecutionContext PD_ExecutionContext;

PADDLE_API int PD_NumInputs(const PD_ExecutionContext* ctx);

PADDLE_API void PD_GetInput(const PD_ExecutionContext* ctx,
                            const std::string& name,
                            PD_Tensor** tensor);

PADDLE_API std::vector<int64_t> PD_GetOutputShape(
    const PD_ExecutionContext* ctx, const std::string& name);

PADDLE_API PD_Tensor* PD_AllocateOutput(const PD_ExecutionContext* ctx,
                                        const std::string& name,
                                        PD_DataType dtype,
                                        const int64_t* dims,
                                        int num_dims);

PADDLE_API void* PD_GetStream(const PD_ExecutionContext* ctx);

PADDLE_API void PD_SetOutput(const PD_ExecutionContext* ctx,
                             const std::string& name,
                             paddle::Tensor& out);

#if defined(_WIN32)
// C-API to get global KernelMetaInfoMap.
__declspec(
    dllexport) inline paddle::KernelMetaInfoMap& PD_GetKernelMetaInfoMap() {
  return paddle::KernelMetaInfoMap::Instance();
}
#endif  // _WIN32

#ifdef __cplusplus
}
#endif
