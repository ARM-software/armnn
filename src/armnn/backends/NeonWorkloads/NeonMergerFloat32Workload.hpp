//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "NeonBaseMergerWorkload.hpp"

namespace armnn
{

class NeonMergerFloat32Workload : public NeonBaseMergerWorkload<DataType::Float16, DataType::Float32>
{
public:
    using NeonBaseMergerWorkload<DataType::Float16, DataType::Float32>::NeonBaseMergerWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
