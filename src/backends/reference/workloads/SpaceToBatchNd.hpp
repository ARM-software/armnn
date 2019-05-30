//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Descriptors.hpp>
#include "armnn/Tensor.hpp"

namespace armnn
{

void SpaceToBatchNd(const TensorInfo& inputInfo,
                    const TensorInfo& outputInfo,
                    const SpaceToBatchNdDescriptor& params,
                    Decoder<float>& inputData,
                    Encoder<float>& outputData);

} //namespace armnn
