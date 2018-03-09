//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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

