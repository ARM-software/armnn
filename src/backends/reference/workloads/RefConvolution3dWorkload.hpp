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

class RefConvolution3dWorkload : public RefBaseWorkload<Convolution3dQueueDescriptor>
{
public:
    explicit RefConvolution3dWorkload(const Convolution3dQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} //namespace armnn

