//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClDepthwiseConvolutionFloat32Workload.hpp"
#include "ClDepthwiseConvolutionHelper.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

ClDepthwiseConvolutionFloat32Workload::ClDepthwiseConvolutionFloat32Workload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : Float32Workload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    InitClDepthwiseConvolutionWorkload(*this);
}

void ClDepthwiseConvolutionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClDepthwiseConvolutionFloat32Workload_Execute");
    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    m_pDepthwiseConvolutionLayer->run();
}

} //namespace armnn
