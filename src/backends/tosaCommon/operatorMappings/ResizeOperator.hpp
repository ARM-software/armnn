//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TosaOperatorUtils.hpp"

using namespace armnn;
using namespace tosa;

TosaSerializationBasicBlock* ConvertResizeToTosaOperator(const Layer* inputSize,
                                                         const std::vector<const TensorInfo*>& outputSize,
                                                         const std::vector<const TensorInfo*>& scale_n,
                                                         const ResizeDescriptor* scale_d);
