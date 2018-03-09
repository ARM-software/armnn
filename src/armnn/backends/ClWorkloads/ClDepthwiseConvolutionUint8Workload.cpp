//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClDepthwiseConvolutionUint8Workload.hpp"
#include "ClDepthwiseConvolutionHelper.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{


ClDepthwiseConvolutionUint8Workload::ClDepthwiseConvolutionUint8Workload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : Uint8Workload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    InitClDepthwiseConvolutionWorkload(*this);
}

void ClDepthwiseConvolutionUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClDepthwiseConvolutionUint8Workload_Execute");
    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    m_pDepthwiseConvolutionLayer->run();
}

} //namespace armnn

