//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClAdditionBaseWorkload.hpp"

namespace armnn
{

class ClAdditionFloatWorkload : public ClAdditionBaseWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClAdditionBaseWorkload<DataType::Float16, DataType::Float32>::ClAdditionBaseWorkload;
    void Execute() const override;
};

} //namespace armnn
