//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Tensor.hpp"

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

void Pad(const TensorInfo& inputInfo,
         const TensorInfo& outputInfo,
         const PadQueueDescriptor& data);

} //namespace armnn
