//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseConstantWorkload.hpp"

namespace armnn
{

class RefConstantFloat32Workload : public RefBaseConstantWorkload<DataType::Float32>
{
public:
    using RefBaseConstantWorkload<DataType::Float32>::RefBaseConstantWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
