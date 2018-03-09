//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/TypesUtils.hpp>
#include "backends/ClLayerSupport.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/ClTensorHandle.hpp"

namespace armnn
{

template <typename WorkloadType>
void InitClDepthwiseConvolutionWorkload(WorkloadType& workload)
{
    using T = typename WorkloadType::KernelDataType;
    using B = typename WorkloadType::BiasDataType;

    auto& m_Data = workload.GetData();
    auto& m_KernelTensor = workload.m_KernelTensor;
    auto& m_BiasTensor = workload.m_BiasTensor;
    auto& m_pDepthwiseConvolutionLayer = workload.m_pDepthwiseConvolutionLayer;

    auto& weightInfo = m_Data.m_Weight->GetTensorInfo();

    std::string reasonIfUnsupported;
    if (!IsClDepthwiseConvolution2dDescParamsSupported(&reasonIfUnsupported, m_Data.m_Parameters, weightInfo))
    {
        throw UnimplementedException(reasonIfUnsupported);
    }

    armcomputetensorutils::BuildArmComputeTensor(m_KernelTensor, weightInfo);

    arm_compute::CLTensor* optionalBias = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        armcomputetensorutils::BuildArmComputeTensor(m_BiasTensor, m_Data.m_Bias->GetTensorInfo());
        optionalBias = &m_BiasTensor;
    }

    arm_compute::PadStrideInfo padStrideInfo(m_Data.m_Parameters.m_StrideX,
                                             m_Data.m_Parameters.m_StrideY,
                                             m_Data.m_Parameters.m_PadLeft,
                                             m_Data.m_Parameters.m_PadRight,
                                             m_Data.m_Parameters.m_PadTop,
                                             m_Data.m_Parameters.m_PadBottom,
                                             arm_compute::DimensionRoundingType::FLOOR);

    std::string name = std::string("ClDepthwiseConvolution") + GetDataTypeName(GetDataType<T>()) + "Workload";
    m_Data.ValidateInputsOutputs(name, 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    //Check for optimisation opportunities.
    bool use3x3Optimisation = (weightInfo.GetShape()[3] == 3) && (weightInfo.GetShape()[2] == 3);
    if (use3x3Optimisation)
    {
        m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::CLDepthwiseConvolutionLayer3x3>();
        static_cast<arm_compute::CLDepthwiseConvolutionLayer3x3*>(m_pDepthwiseConvolutionLayer.get())->configure(
            &input,
            &m_KernelTensor,
            optionalBias,
            &output,
            padStrideInfo);
    }
    else
    {
        m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::CLDepthwiseConvolutionLayer>();
        static_cast<arm_compute::CLDepthwiseConvolutionLayer*>(m_pDepthwiseConvolutionLayer.get())->configure(
            &input,
            &m_KernelTensor,
            optionalBias,
            &output,
            padStrideInfo);
    }

    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    InitialiseArmComputeClTensorData(m_KernelTensor, m_Data.m_Weight->template GetConstTensor<T>());

    if (optionalBias)
    {
        InitialiseArmComputeClTensorData(*optionalBias, m_Data.m_Bias->template GetConstTensor<B>());
    }
}

} //namespace armnn