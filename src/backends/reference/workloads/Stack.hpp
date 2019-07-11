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

void Stack (const StackQueueDescriptor&                   data,
            std::vector<std::unique_ptr<Decoder<float>>>& inputs,
            Encoder<float>&                               output);

} // namespace armnn
