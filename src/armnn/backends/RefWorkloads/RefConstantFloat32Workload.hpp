//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
