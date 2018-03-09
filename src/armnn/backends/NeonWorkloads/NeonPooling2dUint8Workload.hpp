//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/Types.hpp>
#include "NeonPooling2dBaseWorkload.hpp"

namespace armnn
{

class NeonPooling2dUint8Workload : public NeonPooling2dBaseWorkload<armnn::DataType::QuantisedAsymm8>
{
public:
    NeonPooling2dUint8Workload(const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;
};

} //namespace armnn




