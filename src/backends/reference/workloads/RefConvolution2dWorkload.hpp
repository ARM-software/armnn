//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

class RefConvolution2dWorkload : public RefBaseWorkload<Convolution2dQueueDescriptor>
{
public:
    explicit RefConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    const TensorShape m_InputShape;
    const TensorShape m_FilterShape;
    const TensorShape m_OutputShape;
};

} //namespace armnn