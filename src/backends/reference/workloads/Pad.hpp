//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Tensor.hpp"

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

void Pad(const TensorInfo& inputInfo,
         const TensorInfo& outputInfo,
         const ITensorHandle* inputHandle,
         ITensorHandle* outputHandle,
         const PadQueueDescriptor& data);

} //namespace armnn
