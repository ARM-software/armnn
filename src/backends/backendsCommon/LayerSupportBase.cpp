//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Deprecated.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Types.hpp>

#include <backendsCommon/LayerSupportBase.hpp>

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace
{

bool DefaultLayerSupport(const char* func,
                         const char* file,
                         unsigned int line,
                         armnn::Optional<std::string&> reasonIfUnsupported)
{
    // NOTE: We only need to return the reason if the optional parameter is not empty
    if (reasonIfUnsupported)
    {
        std::stringstream message;
        message << func << " is not implemented [" << file << ":" << line << "]";

        reasonIfUnsupported.value() = message.str();
    }

    return false;
}

} // anonymous namespace

namespace armnn
{

bool LayerSupportBase::IsLayerSupported(const LayerType& type,
                                        const std::vector<TensorInfo>& infos,
                                        const BaseDescriptor& descriptor,
                                        const Optional<LstmInputParamsInfo>&,
                                        const Optional<QuantizedLstmInputParamsInfo>&,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    switch(type)
    {
        case LayerType::MemCopy:
            return IsMemCopySupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::MemImport:
            return IsMemImportSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::StandIn:
        {
            auto desc = *(PolymorphicDowncast<const StandInDescriptor*>(&descriptor));

            if (infos.size() != (desc.m_NumInputs + desc.m_NumOutputs))
            {
                throw InvalidArgumentException("Number of StandIn layer TensorInfos does not equal "
                                               "the combined number of input and output slots assigned "
                                               "to the StandIn descriptor");
            }

            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < desc.m_NumInputs; i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            std::vector<const TensorInfo*> outputInfos;
            for (uint32_t i = desc.m_NumInputs; i < infos.size(); i++)
            {
                outputInfos.push_back(&infos[i]);
            }

            return IsStandInSupported(inputInfos,
                                      outputInfos,
                                      desc,
                                      reasonIfUnsupported);
        }
        default:
            return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
    }
}

bool LayerSupportBase::IsActivationSupported(const TensorInfo&, // input
                                             const TensorInfo&, //output
                                             const ActivationDescriptor&, // descriptor
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsAdditionSupported(const TensorInfo&, // input0
                                           const TensorInfo&, // input1
                                           const TensorInfo&, // output
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsArgMinMaxSupported(const armnn::TensorInfo&, // input
                                            const armnn::TensorInfo&, // output
                                            const armnn::ArgMinMaxDescriptor&, // descriptor
                                            armnn::Optional<std::string &> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsBatchNormalizationSupported(const TensorInfo&, //input
                                                     const TensorInfo&, // output
                                                     const TensorInfo&, //mean
                                                     const TensorInfo&, //var
                                                     const TensorInfo&, //beta
                                                     const TensorInfo&, //gamma
                                                     const BatchNormalizationDescriptor&, // descriptor
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsBatchToSpaceNdSupported(const TensorInfo&, // input
                                                 const TensorInfo&, // output
                                                 const BatchToSpaceNdDescriptor&, //descriptor
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsCastSupported(const TensorInfo&, //input
                                       const TensorInfo&, //output
                                       Optional<std::string &> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsChannelShuffleSupported(const TensorInfo&, //input
                                                 const TensorInfo&, //output
                                                 const ChannelShuffleDescriptor&, //descriptor
                                                 Optional<std::string &> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsComparisonSupported(const TensorInfo&, // input0
                                             const TensorInfo&, // input1
                                             const TensorInfo&, // output
                                             const ComparisonDescriptor&, // descriptor
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsConcatSupported(const std::vector<const TensorInfo*>, // inputs
                                         const TensorInfo&, // output
                                         const OriginsDescriptor&, // descriptor
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsConstantSupported(const TensorInfo&, // output
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsConvertFp16ToFp32Supported(const TensorInfo&, // input
                                                    const TensorInfo&, // output
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsConvertFp32ToFp16Supported(const TensorInfo&, // input
                                                    const TensorInfo&, // output
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsConvolution2dSupported(const TensorInfo&, // input
                                                const TensorInfo&, // output
                                                const Convolution2dDescriptor&, // descriptor
                                                const TensorInfo&, // weights
                                                const Optional<TensorInfo>&, // biases
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsConvolution3dSupported(const TensorInfo&, // input
                                                const TensorInfo&, // output
                                                const Convolution3dDescriptor&, // descriptor
                                                const TensorInfo&, // weights
                                                const Optional<TensorInfo>&, // biases
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDebugSupported(const TensorInfo&, // input
                                        const TensorInfo&, // output
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDepthToSpaceSupported(const TensorInfo&, // input
                                               const TensorInfo&, // output
                                               const DepthToSpaceDescriptor&, // descriptor
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDepthwiseConvolutionSupported(const TensorInfo&, //input
                                                       const TensorInfo&, //output
                                                       const DepthwiseConvolution2dDescriptor&, // descriptor
                                                       const TensorInfo&, // weights
                                                       const Optional<TensorInfo>&, // biases
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDequantizeSupported(const TensorInfo&, // input
                                             const TensorInfo&, // output
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDetectionPostProcessSupported(const TensorInfo&, // boxEncodings
                                                       const TensorInfo&, // scores
                                                       const TensorInfo&, // anchors
                                                       const TensorInfo&, // detectionBoxes
                                                       const TensorInfo&, // detectionClasses
                                                       const TensorInfo&, // detectionScores
                                                       const TensorInfo&, // numDetections
                                                       const DetectionPostProcessDescriptor&, //descriptor
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDilatedDepthwiseConvolutionSupported(const TensorInfo&, // input
                                                              const TensorInfo&, // output
                                                              const DepthwiseConvolution2dDescriptor&, // descriptor
                                                              const TensorInfo&,// weights
                                                              const Optional<TensorInfo>&, // biases
                                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsDivisionSupported(const TensorInfo&, // input0
                                           const TensorInfo&, // input1
                                           const TensorInfo&, // output
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsElementwiseUnarySupported(const TensorInfo&, // input
                                                   const TensorInfo&, // output
                                                   const ElementwiseUnaryDescriptor&, // descriptor
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsFakeQuantizationSupported(const TensorInfo&, // input
                                                   const FakeQuantizationDescriptor&, // descriptor
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsFillSupported(const TensorInfo&, // input
                                       const TensorInfo&, // output
                                       const FillDescriptor&, // descriptor
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsFloorSupported(const TensorInfo&, // input
                                        const TensorInfo&, // output
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsFullyConnectedSupported(const TensorInfo&, // input
                                                 const TensorInfo&, // output
                                                 const TensorInfo&, // weights
                                                 const TensorInfo&, // biases
                                                 const FullyConnectedDescriptor&, // descriptor
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsGatherSupported(const armnn::TensorInfo&, // input0
                                         const armnn::TensorInfo&, // input1
                                         const armnn::TensorInfo&, // output
                                         const GatherDescriptor&, // descriptor
                                         armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsInputSupported(const TensorInfo&, // input
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsInstanceNormalizationSupported(const TensorInfo&, // input
                                                        const TensorInfo&, // output
                                                        const InstanceNormalizationDescriptor&, // descriptor
                                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsL2NormalizationSupported(const TensorInfo&, // input
                                                  const TensorInfo&, // output
                                                  const L2NormalizationDescriptor&, // descriptor
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsLogicalBinarySupported(const TensorInfo&, // input0
                                                const TensorInfo&, // input1
                                                const TensorInfo&, // output
                                                const LogicalBinaryDescriptor&, // descriptor
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsLogicalUnarySupported(const TensorInfo&, // input
                                               const TensorInfo&, // output
                                               const ElementwiseUnaryDescriptor&, // descriptor
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsLogSoftmaxSupported(const TensorInfo&, // input
                                             const TensorInfo&, // output
                                             const LogSoftmaxDescriptor&, // descriptor
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsLstmSupported(const TensorInfo&, // input
                                       const TensorInfo&, // outputStateIn
                                       const TensorInfo&, // cellStateIn
                                       const TensorInfo&, // scratchBuffer
                                       const TensorInfo&, // outputStateOut
                                       const TensorInfo&, // cellStateOut
                                       const TensorInfo&, // output
                                       const LstmDescriptor&, // descriptor
                                       const LstmInputParamsInfo&, // paramsInfo
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsMaximumSupported(const TensorInfo&, // input0
                                          const TensorInfo&, // input1
                                          const TensorInfo&, // output
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsMeanSupported(const TensorInfo&, // input
                                       const TensorInfo&, // output
                                       const MeanDescriptor&, // descriptor
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsMemCopySupported(const armnn::TensorInfo&, // input
                                          const armnn::TensorInfo&, // output
                                          armnn::Optional<std::string &> ) const // reasonIfUnsupported
{
    return true;
}

bool LayerSupportBase::IsMemImportSupported(const armnn::TensorInfo&, // input
                                            const armnn::TensorInfo&, // output
                                            armnn::Optional<std::string &> ) const // reasonIfUnsupported
{
    return true;
}

bool LayerSupportBase::IsMergeSupported(const TensorInfo&, // input0
                                        const TensorInfo&, // input1
                                        const TensorInfo&, // output
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsMinimumSupported(const TensorInfo&, // input0
                                          const TensorInfo&, // input1
                                          const TensorInfo&, // output
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsMultiplicationSupported(const TensorInfo&, // input0
                                                 const TensorInfo&, // input1
                                                 const TensorInfo&, // output
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsNormalizationSupported(const TensorInfo&, // input
                                                const TensorInfo&, // output
                                                const NormalizationDescriptor&, // descriptor
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsOutputSupported(const TensorInfo&, // output
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsPadSupported(const TensorInfo&, // input
                                      const TensorInfo&, // output
                                      const PadDescriptor&, // descriptor
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsPermuteSupported(const TensorInfo&, // input
                                          const TensorInfo&, // output
                                          const PermuteDescriptor&, // descriptor
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsPooling2dSupported(const TensorInfo&, // input
                                            const TensorInfo&, // output
                                            const Pooling2dDescriptor&, // descriptor
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsPooling3dSupported(const TensorInfo&, // input
                                            const TensorInfo&, // output
                                            const Pooling3dDescriptor&, // descriptor
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsPreCompiledSupported(const TensorInfo&, // input
                                              const PreCompiledDescriptor&, // descriptor
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsPreluSupported(const TensorInfo&, // input
                                        const TensorInfo&, // alpha
                                        const TensorInfo&, // output
                                        Optional<std::string &> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsQuantizeSupported(const armnn::TensorInfo&, // input
                                           const armnn::TensorInfo&, // output
                                           armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsQLstmSupported(const TensorInfo&, // input
                                        const TensorInfo&, // previousOutputIn
                                        const TensorInfo&, // previousCellStateIn
                                        const TensorInfo&, // outputStateOut
                                        const TensorInfo&, // cellStateOut
                                        const TensorInfo&, // output
                                        const QLstmDescriptor&, // descriptor
                                        const LstmInputParamsInfo&, // paramsInfo
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsQuantizedLstmSupported(const TensorInfo&, // input
                                                const TensorInfo&, // previousCellStateIn
                                                const TensorInfo&, // previousOutputIn
                                                const TensorInfo&, // cellStateOut
                                                const TensorInfo&, // output
                                                const QuantizedLstmInputParamsInfo&, // paramsInfo
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsRankSupported(const TensorInfo&, // input
                                       const TensorInfo&,  // output
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsReduceSupported(const TensorInfo& /*input*/,
                                         const TensorInfo& /*output*/,
                                         const ReduceDescriptor& /*descriptor*/,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsReshapeSupported(const TensorInfo&, // input
                                          const TensorInfo&, // output
                                          const ReshapeDescriptor&, // descriptor
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsResizeSupported(const TensorInfo&, // input
                                         const TensorInfo&, // output
                                         const ResizeDescriptor&, // descriptor
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsShapeSupported(const TensorInfo&, // input
                                        const TensorInfo&, // output
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsSliceSupported(const TensorInfo&, // input
                                        const TensorInfo&, // output
                                        const SliceDescriptor&, // descriptor
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsSoftmaxSupported(const TensorInfo&, // input
                                          const TensorInfo&, // output
                                          const SoftmaxDescriptor&, // descriptor
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}
/**/
bool LayerSupportBase::IsSpaceToBatchNdSupported(const TensorInfo&, // input
                                                 const TensorInfo&, // output
                                                 const SpaceToBatchNdDescriptor&, // descriptor
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsSpaceToDepthSupported(const TensorInfo&, // input
                                               const TensorInfo&, // output
                                               const SpaceToDepthDescriptor&, // descriptor
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsSplitterSupported(const TensorInfo&, // input
                                           const std::vector<std::reference_wrapper<TensorInfo>>&, // outputs
                                           const ViewsDescriptor&, // descriptor
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsStackSupported(const std::vector<const TensorInfo*>&, // inputs
                                        const TensorInfo&, // output
                                        const StackDescriptor&, // descriptor
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsStandInSupported(const std::vector<const TensorInfo*>&, // inputs
                                          const std::vector<const TensorInfo*>&, // outputs
                                          const StandInDescriptor&, // descriptor
                                          Optional<std::string&> reasonIfUnsupported) const
{
    if (reasonIfUnsupported)
    {
        std::stringstream message;
        message << "StandIn layer is not executable via backends";

        reasonIfUnsupported.value() = message.str();
    }

    return false;
}

bool LayerSupportBase::IsStridedSliceSupported(const TensorInfo&, // input
                                               const TensorInfo&, // output
                                               const StridedSliceDescriptor&, // descriptor
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsSubtractionSupported(const TensorInfo&, // input0
                                              const TensorInfo&, // input1
                                              const TensorInfo&, // output
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsSwitchSupported(const TensorInfo&, // input0
                                         const TensorInfo&, // input1
                                         const TensorInfo&, // output0
                                         const TensorInfo&, // output1
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsTransposeConvolution2dSupported(const TensorInfo&, // input
                                                         const TensorInfo&, // output
                                                         const TransposeConvolution2dDescriptor&, // descriptor
                                                         const TensorInfo&, // weights
                                                         const Optional<TensorInfo>&, // biases
                                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
} 

bool LayerSupportBase::IsTransposeSupported(const TensorInfo&, // input 
                                            const TensorInfo&, // output
                                            const TransposeDescriptor&, // descriptor
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool LayerSupportBase::IsUnidirectionalSequenceLstmSupported(const TensorInfo&, // input
                                                             const TensorInfo&, // outputStateIn
                                                             const TensorInfo&, // cellStateIn
                                                             const TensorInfo&, // outputStateOut
                                                             const TensorInfo&, // cellStateOut
                                                             const TensorInfo&, // output
                                                             const LstmDescriptor&, // descriptor
                                                             const LstmInputParamsInfo&, // paramsInfo
                                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

} // namespace armnn
