//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

namespace armnn
{

class RefStridedSliceWorkload : public BaseWorkload<StridedSliceQueueDescriptor>
{
public:
    RefStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& descriptor) override;
};

} // namespace armnn
