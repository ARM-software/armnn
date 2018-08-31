//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ClBaseConstantWorkload.hpp"

namespace armnn
{
class ClConstantFloat32Workload : public ClBaseConstantWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClBaseConstantWorkload<DataType::Float16, DataType::Float32>::ClBaseConstantWorkload;
    void Execute() const override;
};


} //namespace armnn