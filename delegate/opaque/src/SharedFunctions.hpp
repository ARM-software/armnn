//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn_delegate.hpp>

#include <tensorflow/lite/c/builtin_op_data.h>

namespace armnnOpaqueDelegate
{

TfLiteStatus ValidateFloorOperator(DelegateData& delegateData,
                                   TfLiteOpaqueContext* tfLiteContext,
                                   const armnn::TensorInfo& inputTensorInfo,
                                   const armnn::TensorInfo& outputTensorInfo);

TfLiteStatus ValidateFusedActivationOperator(DelegateData& delegateData,
                                             TfLiteOpaqueContext* tfLiteContext,
                                             const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             TfLiteFusedActivation activationType);

} // namespace armnnOpaqueDelegate

