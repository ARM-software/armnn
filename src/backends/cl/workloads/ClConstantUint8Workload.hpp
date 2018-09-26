//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseConstantWorkload.hpp"

namespace armnn
{

class ClConstantUint8Workload : public ClBaseConstantWorkload<DataType::QuantisedAsymm8>
{
public:
    using ClBaseConstantWorkload<DataType::QuantisedAsymm8>::ClBaseConstantWorkload;
    void Execute() const override;
};

} //namespace armnn
