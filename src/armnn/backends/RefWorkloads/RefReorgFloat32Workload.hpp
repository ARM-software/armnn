#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

    class RefReorgFloat32Workload : public Float32Workload<ReorgQueueDescriptor>
    {
    public:
        using Float32Workload<ReorgQueueDescriptor>::Float32Workload;

        virtual void Execute() const override;
    };

} //namespace armnn