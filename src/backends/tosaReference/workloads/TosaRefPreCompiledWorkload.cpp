//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefPreCompiledWorkload.hpp"

namespace armnn
{

TosaRefPreCompiledWorkload::TosaRefPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info)
{
    // Do nothing for now
}

void TosaRefPreCompiledWorkload::Execute() const
{
    // Do nothing for now
}

bool TosaRefPreCompiledWorkloadValidate(std::string*)
{
    return true;
}

}    //namespace armnn
