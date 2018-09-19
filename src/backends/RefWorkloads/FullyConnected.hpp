//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

namespace armnn
{

/// Performs a matrix multiplication and optionally adds a bias.
void FullyConnected(const float*      inputData,
                    float*            outputData,
                    const TensorInfo& inputTensorInfo,
                    const TensorInfo& outputTensorInfo,
                    const float*      weightData,
                    const float*      biasData,
                    bool              transposeWeights);

} //namespace armnn
