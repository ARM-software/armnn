//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Workload.hpp"

#include "backends/NeonTensorHandle.hpp"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include <arm_compute/runtime/SubTensor.h>

#include <boost/cast.hpp>

namespace armnn
{
class Layer;

template<typename T>
void InitialiseArmComputeTensorData(arm_compute::Tensor& tensor, const T* data);

} //namespace armnn
