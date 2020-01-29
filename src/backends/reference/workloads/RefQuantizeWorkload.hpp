//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn {

class RefQuantizeWorkload : public BaseWorkload<QuantizeQueueDescriptor>
{
public:
    RefQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo &info);
    void PostAllocationConfigure() override;
    void Execute() const override;

private:

    std::unique_ptr<Decoder<float>> m_InputDecoder;
    std::unique_ptr<Encoder<float>> m_OutputEncoder;

    size_t m_NumElements;
};

} //namespace armnn