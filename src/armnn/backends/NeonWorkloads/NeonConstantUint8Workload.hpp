//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseConstantWorkload.hpp"

namespace armnn
{

class NeonConstantUint8Workload : public NeonBaseConstantWorkload<DataType::QuantisedAsymm8>
{
public:
    using NeonBaseConstantWorkload<DataType::QuantisedAsymm8>::NeonBaseConstantWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
