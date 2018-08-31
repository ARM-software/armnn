//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

class ClConvertFp16ToFp32Workload : public Float16ToFloat32Workload<ConvertFp16ToFp32QueueDescriptor>
{
public:

    ClConvertFp16ToFp32Workload(const ConvertFp16ToFp32QueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLDepthConvertLayer m_Layer;
};

arm_compute::Status ClConvertFp16ToFp32WorkloadValidate(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        std::string* reasonIfUnsupported);

} //namespace armnn
