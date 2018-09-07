//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClSubtractionBaseWorkload.hpp"

namespace armnn
{

class ClSubtractionFloatWorkload : public ClSubtractionBaseWorkload<DataType::Float16, DataType::Float32>
{
public:
    using ClSubtractionBaseWorkload<DataType::Float16, DataType::Float32>::ClSubtractionBaseWorkload;
    void Execute() const override;
};

} //namespace armnn
