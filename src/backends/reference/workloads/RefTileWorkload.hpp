//
// Copyright © 2020-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>

#include "Tile.hpp"

namespace armnn
{

    class RefTileWorkload : public RefBaseWorkload<TileQueueDescriptor>
    {
    public:
        explicit RefTileWorkload(const TileQueueDescriptor& descriptor,
                                 const WorkloadInfo& info);

        void Execute() const override;

    private:
        template <typename T>
        void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

    };

} // namespace armnn