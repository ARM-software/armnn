//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

namespace armnn
{

class RefDequantizeWorkload : public RefBaseWorkload<DequantizeQueueDescriptor>
{
public:
    using RefBaseWorkload<DequantizeQueueDescriptor>::m_Data;
    using RefBaseWorkload<DequantizeQueueDescriptor>::RefBaseWorkload;

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;
private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} // namespace armnn
