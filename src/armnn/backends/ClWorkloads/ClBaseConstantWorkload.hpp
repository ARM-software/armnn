//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ClWorkloadUtils.hpp"

namespace armnn
{
template <armnn::DataType... DataTypes>
class ClBaseConstantWorkload : public TypedWorkload<ConstantQueueDescriptor, DataTypes...>
{
public:
    ClBaseConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TypedWorkload<ConstantQueueDescriptor, DataTypes...>(descriptor, info)
        , m_RanOnce(false)
    {
    }

    void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn