//
// Copyright © 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Layer.hpp>

#include <tosa_serialization_handler.h>

#include "TosaOperatorUtils.hpp"

using namespace armnn;
using namespace tosa;

TosaSerializationBasicBlock* ConvertPReluToTosaOperator(const Layer* layer,
                                                        const std::vector<const TensorInfo*>& inputs,
                                                        const std::vector<const TensorInfo*>& outputs);