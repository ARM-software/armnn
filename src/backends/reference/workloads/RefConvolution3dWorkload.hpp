//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

class RefConvolution3dWorkload : public RefBaseWorkload<Convolution3dQueueDescriptor>
{
public:
    explicit RefConvolution3dWorkload(const Convolution3dQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);

    void PostAllocationConfigure() override;

    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void PostAllocationConfigure(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs);
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    std::unique_ptr<Decoder<float>> m_FilterDecoder;
    std::unique_ptr<Decoder<float>> m_BiasDecoder;

    TensorShape m_FilterShape;
};

} //namespace armnn

