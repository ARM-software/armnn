//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
