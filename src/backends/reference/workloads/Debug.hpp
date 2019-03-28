//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename T>
void Debug(const TensorInfo& inputInfo,
           const T* inputData,
           LayerGuid guid,
           const std::string& layerName,
           unsigned int slotIndex);

} //namespace armnn
