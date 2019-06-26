//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Decoders.hpp"
#include "Encoders.hpp"

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/Workload.hpp>

namespace armnn
{

class RefTransposeConvolution2dWorkload : public BaseWorkload<TransposeConvolution2dQueueDescriptor>
{
public:
    RefTransposeConvolution2dWorkload(const TransposeConvolution2dQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);
    ~RefTransposeConvolution2dWorkload() = default;

    void PostAllocationConfigure() override;

    void Execute() const override;

private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weights;
    std::unique_ptr<ScopedCpuTensorHandle> m_Biases;

    std::unique_ptr<Decoder<float>> m_InputDecoder;
    std::unique_ptr<Encoder<float>> m_OutputEncoder;

    std::unique_ptr<Decoder<float>> m_WeightsDecoder;
    std::unique_ptr<Decoder<float>> m_BiasesDecoder;

    TensorShape m_InputShape;
    TensorShape m_OutputShape;
    TensorShape m_WeightsShape;
};

} // namespace armnn