//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/ITensorHandle.hpp>

#include <arm_compute/runtime/IMemoryGroup.h>
#include <arm_compute/runtime/Tensor.h>

namespace armnn
{

class IAclTensorHandle : public ITensorHandle
{
public:
    virtual arm_compute::ITensor& GetTensor() = 0;
    virtual arm_compute::ITensor const& GetTensor() const = 0;
    virtual arm_compute::DataType GetDataType() const = 0;
    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) = 0;
};

} //namespace armnn