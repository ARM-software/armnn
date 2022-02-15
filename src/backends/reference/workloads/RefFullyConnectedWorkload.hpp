//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"


namespace armnn
{

class RefFullyConnectedWorkload : public RefBaseWorkload<FullyConnectedQueueDescriptor>
{
public:
    explicit RefFullyConnectedWorkload(const FullyConnectedQueueDescriptor& descriptor,
                                       const WorkloadInfo& info);

    void PostAllocationConfigure() override;

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void PostAllocationConfigure(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs);
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    std::unique_ptr<ScopedTensorHandle> m_Weight;
    std::unique_ptr<ScopedTensorHandle> m_Bias;

    std::unique_ptr<Decoder<float>> m_WeightDecoder;
    std::unique_ptr<Decoder<float>> m_BiasDecoder;

    TensorShape m_InputShape;
    TensorShape m_OutputShape;
    TensorShape m_WeightShape;
    unsigned int m_NumActivations;
};

} //namespace armnn
