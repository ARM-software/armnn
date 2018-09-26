//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backends/Workload.hpp>
#include <boost/optional.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

arm_compute::Status ClDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                           const TensorInfo& output,
                                                           const DepthwiseConvolution2dDescriptor& descriptor,
                                                           const TensorInfo& weights,
                                                           const boost::optional<TensorInfo>& biases);

template<armnn::DataType... dataTypes>
class ClDepthwiseConvolutionBaseWorkload : public TypedWorkload<DepthwiseConvolution2dQueueDescriptor, dataTypes...>
{
public:
    using TypedWorkload<DepthwiseConvolution2dQueueDescriptor, dataTypes...>::m_Data;

    ClDepthwiseConvolutionBaseWorkload(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                       const WorkloadInfo& info);

protected:
    std::unique_ptr<arm_compute::IFunction> m_DepthwiseConvolutionLayer;

    std::unique_ptr<arm_compute::CLTensor> m_KernelTensor;
    std::unique_ptr<arm_compute::CLTensor> m_BiasTensor;

    void FreeUnusedTensors();
};

} //namespace armnn
