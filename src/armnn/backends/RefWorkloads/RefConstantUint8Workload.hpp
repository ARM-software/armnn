//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
