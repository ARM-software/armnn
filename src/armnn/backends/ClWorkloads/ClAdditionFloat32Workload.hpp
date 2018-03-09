//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClAdditionFloat32Workload : public Float32Workload<AdditionQueueDescriptor>
{
public:
    ClAdditionFloat32Workload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

    static bool IsSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported);

private:
    mutable arm_compute::CLArithmeticAddition m_Layer;
    static constexpr arm_compute::ConvertPolicy ms_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;
};

} //namespace armnn