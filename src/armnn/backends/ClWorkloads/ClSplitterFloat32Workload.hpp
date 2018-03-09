//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "ClBaseSplitterWorkload.hpp"

namespace armnn
{

class ClSplitterFloat32Workload : public ClBaseSplitterWorkload<DataType::Float32>
{
public:
    using ClBaseSplitterWorkload<DataType::Float32>::ClBaseSplitterWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
