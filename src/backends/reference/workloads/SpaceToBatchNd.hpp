//
// Copyright © 2017-2019,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Descriptors.hpp>

namespace armnn
{

void SpaceToBatchNd(const TensorInfo& inputInfo,
                    const TensorInfo& outputInfo,
                    const SpaceToBatchNdDescriptor& params,
                    Decoder<float>& inputData,
                    Encoder<float>& outputData);

} //namespace armnn
