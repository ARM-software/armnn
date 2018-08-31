//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/NeonWorkloadUtils.hpp>
#include "NeonPooling2dBaseWorkload.hpp"

namespace armnn
{

class NeonPooling2dFloatWorkload : public NeonPooling2dBaseWorkload<armnn::DataType::Float16,
                                                                    armnn::DataType::Float32>
{
public:
    NeonPooling2dFloatWorkload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
};

} //namespace armnn



