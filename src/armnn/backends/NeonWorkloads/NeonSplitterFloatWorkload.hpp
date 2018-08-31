//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "NeonBaseSplitterWorkload.hpp"

namespace armnn
{

class NeonSplitterFloatWorkload : public NeonBaseSplitterWorkload<DataType::Float16, DataType::Float32>
{
public:
    using NeonBaseSplitterWorkload<DataType::Float16, DataType::Float32>::NeonBaseSplitterWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
