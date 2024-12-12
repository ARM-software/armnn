//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"

namespace armnn
{
class RefBroadcastToWorkload : public RefBaseWorkload<BroadcastToQueueDescriptor>
{

public:
    explicit RefBroadcastToWorkload(const BroadcastToQueueDescriptor& descriptor,
                                    const WorkloadInfo& info);

    void Execute() const override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;
};
} // namespace armnn