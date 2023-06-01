//
// Copyright © 2018-2019,2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

namespace armnn
{

class RefBatchToSpaceNdWorkload : public RefBaseWorkload<BatchToSpaceNdQueueDescriptor>
{

public:
    using RefBaseWorkload<BatchToSpaceNdQueueDescriptor>::RefBaseWorkload;

    void Execute() const override;
    void ExecuteAsync(ExecutionData& executionData)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};

} // namespace armnn