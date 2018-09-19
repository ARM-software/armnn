//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseMergerWorkload.hpp"

namespace armnn
{

class ClMergerFloatWorkload : public ClBaseMergerWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClBaseMergerWorkload<DataType::Float16, DataType::Float32>::ClBaseMergerWorkload;
    virtual void Execute() const override;
};

} //namespace armnn


