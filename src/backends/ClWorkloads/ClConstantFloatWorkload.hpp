//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseConstantWorkload.hpp"

namespace armnn
{
class ClConstantFloatWorkload : public ClBaseConstantWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClBaseConstantWorkload<DataType::Float16, DataType::Float32>::ClBaseConstantWorkload;
    void Execute() const override;
};


} //namespace armnn
