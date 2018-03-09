//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClConvolution2dUint8Workload.hpp"
#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/ClLayerSupport.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClConvolution2dUint8Workload::ClConvolution2dUint8Workload(const Convolution2dQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info)
    : Uint8Workload<Convolution2dQueueDescriptor>(descriptor, info)
{

    // todo: check tensor shapes match
    const TensorInfo& weightInfo = m_Data.m_Weight->GetTensorInfo();
    BuildArmComputeTensor(m_KernelTensor, weightInfo);

    arm_compute::PadStrideInfo padStrideInfo(m_Data.m_Parameters.m_StrideX,
                                             m_Data.m_Parameters.m_StrideY,
                                             m_Data.m_Parameters.m_PadLeft,
                                             m_Data.m_Parameters.m_PadRight,
                                             m_Data.m_Parameters.m_PadTop,
                                             m_Data.m_Parameters.m_PadBottom,
                                             arm_compute::DimensionRoundingType::FLOOR);

    arm_compute::CLTensor* optionalBias = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        BuildArmComputeTensor(m_BiasTensor, m_Data.m_Bias->GetTensorInfo());
        optionalBias = &m_BiasTensor;
    }

    m_Data.ValidateInputsOutputs("ClConvolution2dUint8Workload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    BOOST_ASSERT_MSG(IsClDirectConvolution2dSupported(weightInfo, m_Data.m_Parameters),
                     "Unsupported parameters for u8 convolution");

    m_pConvolutionLayer = std::make_unique<arm_compute::CLDirectConvolutionLayer>();
    static_cast<arm_compute::CLDirectConvolutionLayer*>(m_pConvolutionLayer.get())->configure(&input,
                                                                                              &m_KernelTensor,
                                                                                              optionalBias,
                                                                                              &output,
                                                                                              padStrideInfo);
    BOOST_ASSERT(m_pConvolutionLayer);

    InitialiseArmComputeClTensorData(m_KernelTensor, m_Data.m_Weight->GetConstTensor<uint8_t>());

    if (optionalBias)
    {
        InitialiseArmComputeClTensorData(*optionalBias, m_Data.m_Bias->GetConstTensor<int32_t>());
    }
}

void ClConvolution2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClConvolution2dUint8Workload_Execute");
    BOOST_ASSERT(m_pConvolutionLayer);

    m_pConvolutionLayer->run();
}

} //namespace armnn
