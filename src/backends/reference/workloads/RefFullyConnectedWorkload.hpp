//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    const TensorShape m_InputShape;
    const TensorShape m_WeightShape;
    const TensorShape m_OutputShape;
    const unsigned int m_NumActivations;
};

} //namespace armnn
