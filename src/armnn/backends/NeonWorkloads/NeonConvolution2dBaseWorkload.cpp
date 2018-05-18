//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/NeonLayerSupport.hpp"

#include "NeonConvolution2dBaseWorkload.hpp"

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonConvolution2dWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const Convolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const TensorInfo& biases)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights);
    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        aclBiasesInfo = BuildArmComputeTensorInfo(biases);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    return arm_compute::NEConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo);
}

template<armnn::DataType dataType>
NeonConvolution2dBaseWorkload<dataType>::NeonConvolution2dBaseWorkload(const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : TypedWorkload<Convolution2dQueueDescriptor, dataType>(descriptor, info)
{
    using arm_compute::NEDirectConvolutionLayer;
    using namespace armcomputetensorutils;

    ValidateData();

    // todo: check tensor shapes match

    arm_compute::ITensor& input = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    BuildArmComputeTensor(m_KernelTensor, m_Data.m_Weight->GetTensorInfo());

    arm_compute::Tensor* optionalBiasTensor = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        BuildArmComputeTensor(m_BiasTensor, m_Data.m_Bias->GetTensorInfo());
        optionalBiasTensor = &m_BiasTensor;
    }

    arm_compute::PadStrideInfo padStrideInfo(m_Data.m_Parameters.m_StrideX,
                                             m_Data.m_Parameters.m_StrideY,
                                             m_Data.m_Parameters.m_PadLeft,
                                             m_Data.m_Parameters.m_PadRight,
                                             m_Data.m_Parameters.m_PadTop,
                                             m_Data.m_Parameters.m_PadBottom,
                                             arm_compute::DimensionRoundingType::FLOOR);

    const bool preferDirectConvolution =
        IsNeonDirectConvolutionPreferred(m_Data.m_Weight->GetTensorInfo(),
                                         m_Data.m_Parameters);

    if (preferDirectConvolution)
    {
        auto directConvolutionLayer = std::make_unique<arm_compute::NEDirectConvolutionLayer>(memoryManager);
        directConvolutionLayer->configure(&input,
                                          &m_KernelTensor,
                                          optionalBiasTensor,
                                          &output,
                                          padStrideInfo);
        m_ConvolutionLayer.reset(directConvolutionLayer.release());
    }
    else
    {
        auto convolutionLayer = std::make_unique<arm_compute::NEConvolutionLayer>(memoryManager);
        convolutionLayer->configure(&input,
                                    &m_KernelTensor,
                                    optionalBiasTensor,
                                    &output,
                                    padStrideInfo);
        m_ConvolutionLayer.reset(convolutionLayer.release());
    }
    BOOST_ASSERT(m_ConvolutionLayer);

    using Type = ResolveType<dataType>;

    InitialiseArmComputeTensorData(m_KernelTensor, m_Data.m_Weight->template GetConstTensor<Type>());
}

// Generate known implementations for linker
template class NeonConvolution2dBaseWorkload<DataType::Float32>;
template class NeonConvolution2dBaseWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn

