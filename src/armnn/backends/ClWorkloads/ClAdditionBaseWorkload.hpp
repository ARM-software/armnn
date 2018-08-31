//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{

template <armnn::DataType... dataTypes>
class ClAdditionBaseWorkload : public TypedWorkload<AdditionQueueDescriptor, dataTypes...>
{
public:
    ClAdditionBaseWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLArithmeticAddition m_Layer;
};

bool ClAdditionValidate(const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        std::string* reasonIfUnsupported);
} //namespace armnn
