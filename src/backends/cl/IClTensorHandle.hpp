//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <arm_compute/core/CL/ICLTensor.h>
#include <arm_compute/runtime/MemoryGroup.h>

namespace armnn
{

class IClTensorHandle : public IAclTensorHandle
{
public:
    virtual arm_compute::ICLTensor& GetTensor() = 0;
    virtual arm_compute::ICLTensor const& GetTensor() const = 0;
    virtual arm_compute::DataType GetDataType() const = 0;
    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) = 0;
};

} //namespace armnn