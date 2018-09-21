//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Workload.hpp"

#include "backends/NeonTensorHandle.hpp"
#include "NeonTimer.hpp"

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

void InitializeArmComputeTensorDataForFloatTypes(arm_compute::Tensor& tensor, const ConstCpuTensorHandle* handle);
} //namespace armnn


#define     ARMNN_SCOPED_PROFILING_EVENT_NEON(name) \
    ARMNN_SCOPED_PROFILING_EVENT_WITH_INSTRUMENTS(armnn::Compute::CpuAcc, \
                                                  name, \
                                                  armnn::NeonTimer(), \
                                                  armnn::WallClockTimer())
