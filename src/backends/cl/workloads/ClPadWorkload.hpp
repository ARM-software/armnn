//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/WorkloadData.hpp>
#include <backends/Workload.hpp>
#include <arm_compute/runtime/CL/functions/CLPadLayer.h>

namespace armnn {

template <armnn::DataType... dataTypes>
class ClPadWorkload : public TypedWorkload<PadQueueDescriptor, dataTypes...>
{
public:
    ClPadWorkload(const PadQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLPadLayer m_Layer;
};

bool ClPadValidate(const TensorInfo& input,
                   const TensorInfo& output,
                   const PadDescriptor& descriptor,
                   std::string* reasonIfUnsupported);

} //namespace armnn

