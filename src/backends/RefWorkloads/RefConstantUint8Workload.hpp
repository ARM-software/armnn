//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseConstantWorkload.hpp"

namespace armnn
{

class RefConstantUint8Workload : public RefBaseConstantWorkload<DataType::QuantisedAsymm8>
{
public:
    using RefBaseConstantWorkload<DataType::QuantisedAsymm8>::RefBaseConstantWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
