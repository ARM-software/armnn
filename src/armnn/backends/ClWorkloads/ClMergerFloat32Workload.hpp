//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ClBaseMergerWorkload.hpp"

namespace armnn
{

class ClMergerFloat32Workload : public ClBaseMergerWorkload<armnn::DataType::Float32>
{
public:
    using ClBaseMergerWorkload<armnn::DataType::Float32>::ClBaseMergerWorkload;
    virtual void Execute() const override;
};

} //namespace armnn


