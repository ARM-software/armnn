//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn_delegate.hpp>

namespace armnnDelegate
{

TfLiteStatus ValidateFloorOperator(DelegateData& delegateData,
                                   TfLiteContext* tfLiteContext,
                                   const armnn::TensorInfo& inputTensorInfo,
                                   const armnn::TensorInfo& outputTensorInfo);

TfLiteStatus ValidateFusedActivationOperator(DelegateData& delegateData,
                                             TfLiteContext* tfLiteContext,
                                             const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             TfLiteFusedActivation activationType);

} // namespace armnnDelegate

