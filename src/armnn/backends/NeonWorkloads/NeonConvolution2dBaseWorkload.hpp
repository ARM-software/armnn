//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/NeonLayerSupport.hpp"
#include "backends/NeonWorkloadUtils.hpp"
#include "backends/Workload.hpp"

#include "arm_compute/runtime/MemoryManagerOnDemand.h"

#include <boost/optional.hpp>

#include <memory>

namespace armnn
{

arm_compute::Status NeonConvolution2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Convolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const boost::optional<TensorInfo>& biases);

template<armnn::DataType... dataTypes>
class NeonConvolution2dBaseWorkload : public TypedWorkload<Convolution2dQueueDescriptor, dataTypes...>
{
public:
    using TypedWorkload<Convolution2dQueueDescriptor, dataTypes...>::m_Data;

    NeonConvolution2dBaseWorkload(const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info,
                                  std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager);

    virtual void ValidateData() const {};

protected:
    std::unique_ptr<arm_compute::IFunction> m_ConvolutionLayer;

    std::unique_ptr<arm_compute::Tensor> m_KernelTensor;
    std::unique_ptr<arm_compute::Tensor> m_BiasTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
