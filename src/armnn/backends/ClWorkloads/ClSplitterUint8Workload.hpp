//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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



