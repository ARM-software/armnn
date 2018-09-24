//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseSplitterWorkload.hpp"

namespace armnn
{

class NeonSplitterUint8Workload : public NeonBaseSplitterWorkload<DataType::QuantisedAsymm8>
{
public:
    using NeonBaseSplitterWorkload<DataType::QuantisedAsymm8>::NeonBaseSplitterWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
