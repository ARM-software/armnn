//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TosaOperatorUtils.hpp"

using namespace armnn;
using namespace tosa;

TosaSerializationBasicBlock* ConvertConstantToTosaOperator(const Layer* layer,
                                                           const std::vector<const TensorInfo*>& outputs);

