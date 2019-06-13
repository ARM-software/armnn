//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Encoders.hpp"
#include "Decoders.hpp"

#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

void PreluImpl(const PreluQueueDescriptor& data,
               Decoder<float>& inputData,
               Decoder<float>& alphaData,
               Encoder<float>& outputData);

} // namespace armnn
