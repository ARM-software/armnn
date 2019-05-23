//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class RefDepthwiseConvolution2dWorkload : public BaseWorkload<DepthwiseConvolution2dQueueDescriptor> {
public:
    explicit RefDepthwiseConvolution2dWorkload(const DepthwiseConvolution2dQueueDescriptor &descriptor,
                                               const WorkloadInfo &info);

    void PostAllocationConfigure() override;

    virtual void Execute() const override;

private:

    std::unique_ptr <ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr <ScopedCpuTensorHandle> m_Bias;

    std::unique_ptr <Decoder<float>> m_InputDecoder;
    std::unique_ptr <Encoder<float>> m_OutputEncoder;
    std::unique_ptr <Decoder<float>> m_FilterDecoder;
    std::unique_ptr <Decoder<float>> m_BiasDecoder;

    TensorShape m_InputShape;
    TensorShape m_OutputShape;
    TensorShape m_FilterShape;
};

} //namespace armnn
