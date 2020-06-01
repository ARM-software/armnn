//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

class RefSplitterWorkload : public BaseWorkload<SplitterQueueDescriptor>
{
public:
    using BaseWorkload<SplitterQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} //namespace armnn
