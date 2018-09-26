//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseMergerWorkload.hpp"

namespace armnn
{

class ClMergerUint8Workload : public ClBaseMergerWorkload<armnn::DataType::QuantisedAsymm8>
{
public:
    using ClBaseMergerWorkload<armnn::DataType::QuantisedAsymm8>::ClBaseMergerWorkload;
    virtual void Execute() const override;
};

} //namespace armnn

