
#pragma once

#include "backends/Workload.hpp"
#include "backends/WorkloadData.hpp"

namespace armnn
{

    class RefDetectionOutputFloat32Workload : public Float32Workload<DetectionOutputQueueDescriptor>
    {
    public:
        using Float32Workload<DetectionOutputQueueDescriptor>::Float32Workload;
        virtual void Execute() const override;
    };

} //namespace armnn