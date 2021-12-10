//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Tensor.hpp"

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

void MirrorPad(const TensorInfo& inputInfo,
               const TensorInfo& outputInfo,
               const ITensorHandle* inputHandle,
               ITensorHandle* outputHandle,
               const PadQueueDescriptor& data);

} //namespace armnn
