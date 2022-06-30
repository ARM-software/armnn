//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/backends/TensorHandle.hpp>
#include "RefBaseWorkload.hpp"

namespace armnn
{

class RefTransposeConvolution2dWorkload : public RefBaseWorkload<TransposeConvolution2dQueueDescriptor>
{
public:
    RefTransposeConvolution2dWorkload(const TransposeConvolution2dQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);
    ~RefTransposeConvolution2dWorkload() = default;

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
    std::unique_ptr<ScopedTensorHandle> m_Weights;
    std::unique_ptr<ScopedTensorHandle> m_Biases;

    std::unique_ptr<Decoder<float>> m_WeightsDecoder;
    std::unique_ptr<Decoder<float>> m_BiasesDecoder;

    TensorShape m_WeightsShape;
};

} // namespace armnn