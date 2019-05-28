//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
};

} // namespace armnn
