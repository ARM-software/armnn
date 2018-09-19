//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseMergerWorkload.hpp"

namespace armnn
{

class NeonMergerUint8Workload : public NeonBaseMergerWorkload<DataType::QuantisedAsymm8>
{
public:
    using NeonBaseMergerWorkload<DataType::QuantisedAsymm8>::NeonBaseMergerWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
