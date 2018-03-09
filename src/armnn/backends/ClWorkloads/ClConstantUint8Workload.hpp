//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
