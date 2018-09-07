//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClSubtractionBaseWorkload.hpp"

namespace armnn
{

class ClSubtractionUint8Workload : public ClSubtractionBaseWorkload<DataType::QuantisedAsymm8>
{
public:
    using ClSubtractionBaseWorkload<DataType::QuantisedAsymm8>::ClSubtractionBaseWorkload;
    void Execute() const override;
};

} //namespace armnn
