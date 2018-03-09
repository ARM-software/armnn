//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

#include <armnn/Types.hpp>

namespace armnn
{

// Base class template providing an implementation of the Constant layer common to all data types
template <armnn::DataType DataType>
class RefBaseConstantWorkload : public TypedWorkload<ConstantQueueDescriptor, DataType>
{
public:
    RefBaseConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TypedWorkload<ConstantQueueDescriptor, DataType>(descriptor, info)
        , m_RanOnce(false)
    {
    }

    virtual void Execute() const override;

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
