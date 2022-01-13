/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <ostream>
#include <string>
#include <tuple>

#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace pten {

// tuple(input_names, attr_names, output_names)
using KernelArgsTuple = std::tuple<paddle::SmallVector<std::string>,
                                   paddle::SmallVector<std::string>,
                                   paddle::SmallVector<std::string>>;

// TODO(chenweihang): Add more methods if needed in future
class ArgumentMappingContext {
 public:
  virtual ~ArgumentMappingContext() = default;

  virtual bool HasInput(const std::string& name) const = 0;
  virtual bool HasOutput(const std::string& name) const = 0;
  virtual bool HasAttr(const std::string& name) const = 0;

  virtual size_t InputSize(const std::string& name) const = 0;
  virtual size_t OutputSize(const std::string& name) const = 0;

  virtual bool IsDenseTensorInput(const std::string& name) const = 0;
  virtual bool IsSelectedRowsInput(const std::string& name) const = 0;
};

struct KernelSignature {
  std::string name;
  KernelArgsTuple args;

  KernelSignature() = default;
  KernelSignature(std::string&& kernel_name,
                  paddle::SmallVector<std::string>&& inputs,
                  paddle::SmallVector<std::string>&& attrs,
                  paddle::SmallVector<std::string>&& outputs)
      : name(std::move(kernel_name)),
        args(std::make_tuple(inputs, attrs, outputs)) {}
  KernelSignature(const std::string& kernel_name,
                  const paddle::SmallVector<std::string>& inputs,
                  const paddle::SmallVector<std::string>& attrs,
                  const paddle::SmallVector<std::string>& outputs)
      : name(kernel_name), args(std::make_tuple(inputs, attrs, outputs)) {}
};

std::ostream& operator<<(std::ostream& os, KernelSignature signature);

using ArgumentMappingFn = KernelSignature (*)(const ArgumentMappingContext&);

class OpArgumentMappingFnMap {
 public:
  static OpArgumentMappingFnMap& Instance();

  bool Has(const std::string& op_type) const;

  const ArgumentMappingFn& Get(const std::string& op_type) const;

  void Emplace(const std::string& op_type,
               const std::string api_name,
               ArgumentMappingFn fn);

 private:
  paddle::flat_hash_map<std::string, std::string> name_map_;
  paddle::flat_hash_map<std::string, ArgumentMappingFn> fn_map_;
};

}  // namespace pten
