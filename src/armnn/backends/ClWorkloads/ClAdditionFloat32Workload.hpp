//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ClAdditionBaseWorkload.hpp"

namespace armnn
{

class ClAdditionFloat32Workload : public ClAdditionBaseWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClAdditionBaseWorkload<DataType::Float16, DataType::Float32>::ClAdditionBaseWorkload;
    void Execute() const override;
};

} //namespace armnn
