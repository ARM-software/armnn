//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseSplitterWorkload.hpp"

namespace armnn
{

class ClSplitterFloatWorkload : public ClBaseSplitterWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClBaseSplitterWorkload<DataType::Float16, DataType::Float32>::ClBaseSplitterWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
