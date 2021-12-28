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

#include "paddle/fluid/framework/custom_kernel.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/kernel_meta_info_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/pten/api/all.h"
#include "paddle/pten/api/lib/api_declare.h"
#include "paddle/pten/api/lib/ext_compat_utils.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace framework {

namespace detail {

// dynamic lib load func
template <typename T>
static T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func, platform::errors::NotFound(
                "Failed to load dynamic operator library, error message(%s).",
                errorno));
  return func;
}

}  // namespace detail

// custom op kernel call function define
static void RunCustomKernelFunc(const framework::ExecutionContext& ctx,
                                const paddle::CustomKernelFunc& func) {
  VLOG(4) << "Custom Kernel: Run ComputeFunc.";
  try {
    func(reinterpret_cast<const PD_ExecutionContext*>(&ctx));
  } catch (platform::EnforceNotMet& exception) {
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(platform::errors::External("%s", ex.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Custom kernel raises an unknown exception in rumtime."));
  }
}

//////////// Kernel Register //////////////

void RegisterCustomKernel(const std::string& name,
                          const pten::KernelKey& kernel_key,
                          const paddle::CustomKernelFunc& kernel_func) {
  auto key = TransPtenKernelKeyToOpKernelType(kernel_key);
  VLOG(1) << "Custom Kernel: op kernel name: " << name << " key: " << key;
  OperatorWithKernel::AllOpKernels()[name][key] =
      [kernel_func](const framework::ExecutionContext& ctx) {
        VLOG(1) << "Custom Kernel: run custom kernel func in lambda.";
        RunCustomKernelFunc(ctx, kernel_func);
      };
}

void RegisterKernelWithMetaInfo(
    const std::vector<KernelMetaInfo>& kernel_meta_infos) {
  for (size_t i = 0; i < kernel_meta_infos.size(); ++i) {
    auto& meta_info = kernel_meta_infos[i];
    auto op_name = KernelMetaInfoHelper::GetOpName(meta_info);
    // check op exists
    if (!OpInfoMap::Instance().Has(op_name)) {
      LOG(WARNING) << "Operator (" << op_name << ") not exsits.";
      return;
    }
    // kernel infomation
    auto kernel_key = KernelMetaInfoHelper::GetKernelKey(meta_info);
    auto& kernel_fn = KernelMetaInfoHelper::GetKernelFn(meta_info);
    RegisterCustomKernel(op_name, kernel_key, kernel_fn);
  }
}

void RegisterKernelWithMetaInfoMap(
    const paddle::KernelMetaInfoMap& kernel_meta_info_map) {
  auto& meta_info_map = kernel_meta_info_map.GetMap();
  VLOG(1) << "Custom Kernel: size of kernel meta info map - "
          << meta_info_map.size();
  // pair: {op_type, KernelMetaInfo}
  for (auto& pair : meta_info_map) {
    VLOG(1) << "Custom Kernel: pair first -> op name: " << pair.first;
    RegisterKernelWithMetaInfo(pair.second);
  }
}

////////////////////// User APIs ///////////////////////

// load kernel api
void LoadKernelMetaInfoAndRegisterKernel(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetKernelDsoHandle(dso_name);
  VLOG(1) << "load custom_kernel lib: " << dso_name;
  typedef KernelMetaInfoMap& get_kernel_meta_info_map_t();
  auto* get_kernel_meta_info_map = detail::DynLoad<get_kernel_meta_info_map_t>(
      handle, "PD_GetKernelMetaInfoMap");
  auto& kernel_meta_info_map = get_kernel_meta_info_map();

  RegisterKernelWithMetaInfoMap(kernel_meta_info_map);
}

}  // namespace framework
}  // namespace paddle
