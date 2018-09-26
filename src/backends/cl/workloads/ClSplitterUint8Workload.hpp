//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseSplitterWorkload.hpp"

namespace armnn
{
class ClSplitterUint8Workload : public ClBaseSplitterWorkload<DataType::QuantisedAsymm8>
{
public:
    using ClBaseSplitterWorkload<DataType::QuantisedAsymm8>::ClBaseSplitterWorkload;
    virtual void Execute() const override;
};
} //namespace armnn



