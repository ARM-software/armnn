//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

ARMNN_NO_DEPRECATE_WARN_BEGIN
// IsLayerSupported() forwards to the deprecated virtual methods depending on input LayerType.
// Allows backends continue to behave as before maintaining backward compatibility.
bool ILayerSupport::IsLayerSupported(const LayerType& type,
                                     const std::vector<TensorInfo>& infos,
                                     const BaseDescriptor& descriptor,
                                     const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                     const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    switch (type)
    {
        case LayerType::Activation:
            return IsActivationSupported(infos[0],
                                         infos[1],
                                         *(PolymorphicDowncast<const ActivationDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        case LayerType::Addition:
            return IsAdditionSupported(infos[0],
                                       infos[1],
                                       infos[2],
                                       reasonIfUnsupported);
        case LayerType::ArgMinMax:
            return IsArgMinMaxSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const ArgMinMaxDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::BatchNormalization:
            return IsBatchNormalizationSupported(infos[0],
                                                 infos[1],
                                                 infos[2],
                                                 infos[3],
                                                 infos[4],
                                                 infos[5],
                                                 *(PolymorphicDowncast<const BatchNormalizationDescriptor*>
                                                     (&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::BatchToSpaceNd:
            return IsBatchToSpaceNdSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const BatchToSpaceNdDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Comparison:
        {
            return IsComparisonSupported(infos[0],
                                         infos[1],
                                         infos[2],
                                         *(PolymorphicDowncast<const ComparisonDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        }
        case LayerType::Concat:
        {
            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < (infos.size() - 1); i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            return IsConcatSupported(inputInfos,
                                     infos[infos.size() - 1],
                                     *(PolymorphicDowncast<const OriginsDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        }
        case LayerType::Constant:
            return IsConstantSupported(infos[0],
                                       reasonIfUnsupported);
        case LayerType::ConvertFp16ToFp32:
            return IsConvertFp16ToFp32Supported(infos[0],
                                                infos[1],
                                                reasonIfUnsupported);
        case LayerType::ConvertFp32ToFp16:
            return IsConvertFp32ToFp16Supported(infos[0],
                                                infos[1],
                                                reasonIfUnsupported);
        case LayerType::Convolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of Convolution2d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsConvolution2dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                EmptyOptional(),
                                                reasonIfUnsupported);
            }
            else
            {
                return IsConvolution2dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                infos[3],
                                                reasonIfUnsupported);
            }
        }
        case LayerType::Debug:
            return IsDebugSupported(infos[0],
                                    infos[1],
                                    reasonIfUnsupported);
        case LayerType::DepthToSpace:
            return IsDepthToSpaceSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const DepthToSpaceDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::DepthwiseConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of DepthwiseConvolution2d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const DepthwiseConvolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsDepthwiseConvolutionSupported(infos[0],
                                                       infos[1],
                                                       desc,
                                                       infos[2],
                                                       EmptyOptional(),
                                                       reasonIfUnsupported);
            }
            else
            {
                return IsDepthwiseConvolutionSupported(infos[0],
                                                       infos[1],
                                                       desc,
                                                       infos[2],
                                                       infos[3],
                                                       reasonIfUnsupported);
            }
        }
        case LayerType::Dequantize:
            return IsDequantizeSupported(infos[0],
                                         infos[1],
                                         reasonIfUnsupported);
        case LayerType::DetectionPostProcess:
            return IsDetectionPostProcessSupported(infos[0],
                                                   infos[1],
                                                   infos[2],
                                                   infos[3],
                                                   infos[4],
                                                   infos[5],
                                                   infos[6],
                                                   *(PolymorphicDowncast<const DetectionPostProcessDescriptor*>
                                                       (&descriptor)),
                                                   reasonIfUnsupported);
        case LayerType::Division:
            return IsDivisionSupported(infos[0],
                                       infos[1],
                                       infos[2],
                                       reasonIfUnsupported);
        case LayerType::ElementwiseUnary:
            return IsElementwiseUnarySupported(infos[0],
                                               infos[1],
                                               *(PolymorphicDowncast<const ElementwiseUnaryDescriptor*>
                                                   (&descriptor)),
                                               reasonIfUnsupported);
        case LayerType::FakeQuantization:
            return IsFakeQuantizationSupported(infos[0],
                                               *(PolymorphicDowncast<const FakeQuantizationDescriptor*>
                                                   (&descriptor)),
                                               reasonIfUnsupported);
        case LayerType::Fill:
            return IsFillSupported(infos[0],
                                   infos[1],
                                   *(PolymorphicDowncast<const FillDescriptor*>(&descriptor)),
                                   reasonIfUnsupported);
        case LayerType::Floor:
            return IsFloorSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::FullyConnected:
            return IsFullyConnectedSupported(infos[0],
                                             infos[1],
                                             infos[2],
                                             infos[3],
                                             *(PolymorphicDowncast<const FullyConnectedDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Gather:
            return IsGatherSupported(infos[0],
                                     infos[1],
                                     infos[2],
                                     *(PolymorphicDowncast<const GatherDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        case LayerType::Input:
            return IsInputSupported(infos[0], reasonIfUnsupported);
        case LayerType::InstanceNormalization:
            return IsInstanceNormalizationSupported(infos[0],
                                                    infos[1],
                                                    *(PolymorphicDowncast<const InstanceNormalizationDescriptor*>
                                                        (&descriptor)),
                                                    reasonIfUnsupported);
        case LayerType::L2Normalization:
            return IsL2NormalizationSupported(infos[0],
                                              infos[1],
                                              *(PolymorphicDowncast<const L2NormalizationDescriptor*>
                                                  (&descriptor)),
                                              reasonIfUnsupported);
        case LayerType::LogicalBinary:
            return IsLogicalBinarySupported(infos[0],
                                            infos[1],
                                            infos[2],
                                            *(PolymorphicDowncast<const LogicalBinaryDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::LogSoftmax:
            return IsLogSoftmaxSupported(infos[0],
                                         infos[1],
                                         *(PolymorphicDowncast<const LogSoftmaxDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        case LayerType::Lstm:
            return IsLstmSupported(infos[0],
                                   infos[1],
                                   infos[2],
                                   infos[3],
                                   infos[4],
                                   infos[5],
                                   infos[6],
                                   *(PolymorphicDowncast<const LstmDescriptor*>(&descriptor)),
                                   lstmParamsInfo.value(),
                                   reasonIfUnsupported);
        case LayerType::QLstm:
            return IsQLstmSupported(infos[0],
                                    infos[1],
                                    infos[2],
                                    infos[3],
                                    infos[4],
                                    infos[5],
                                    *(PolymorphicDowncast<const QLstmDescriptor*>(&descriptor)),
                                    lstmParamsInfo.value(),
                                    reasonIfUnsupported);
        case LayerType::Map:
            return true;
        case LayerType::Maximum:
            return IsMaximumSupported(infos[0],
                                      infos[1],
                                      infos[2],
                                      reasonIfUnsupported);
        case LayerType::Mean:
            return IsMeanSupported(infos[0],
                                   infos[1],
                                   *(PolymorphicDowncast<const MeanDescriptor*>(&descriptor)),
                                   reasonIfUnsupported);
        case LayerType::MemCopy:
            return IsMemCopySupported(std::move(infos[0]),
                                      std::move(infos[1]),
                                      reasonIfUnsupported);
        case LayerType::MemImport:
            return IsMemImportSupported(infos[0],
                                        infos[1],
                                        reasonIfUnsupported);
        case LayerType::Merge:
            return IsMergeSupported(infos[0],
                                    infos[1],
                                    infos[2],
                                    reasonIfUnsupported);
        case LayerType::Minimum:
            return IsMinimumSupported(infos[0],
                                      infos[1],
                                      infos[2],
                                      reasonIfUnsupported);
        case LayerType::Multiplication:
            return IsMultiplicationSupported(infos[0],
                                             infos[1],
                                             infos[2],
                                             reasonIfUnsupported);
        case LayerType::Normalization:
            return IsNormalizationSupported(infos[0],
                                            infos[1],
                                            *(PolymorphicDowncast<const NormalizationDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::Output:
            return IsOutputSupported(infos[0], reasonIfUnsupported);
        case LayerType::Pad:
            return IsPadSupported(infos[0],
                                  infos[1],
                                  *(PolymorphicDowncast<const PadDescriptor*>(&descriptor)),
                                  reasonIfUnsupported);
        case LayerType::Permute:
            return IsPermuteSupported(infos[0],
                                      infos[1],
                                      *(PolymorphicDowncast<const PermuteDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::Pooling2d:
            return IsPooling2dSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::PreCompiled:
            return IsPreCompiledSupported(infos[0],
                                          *(PolymorphicDowncast<const PreCompiledDescriptor*>(&descriptor)),
                                          reasonIfUnsupported);
        case LayerType::Prelu:
            return IsPreluSupported(infos[0],
                                    infos[1],
                                    infos[2],
                                    reasonIfUnsupported);
        case LayerType::Quantize:
            return IsQuantizeSupported(infos[0],
                                       infos[1],
                                       reasonIfUnsupported);
        case LayerType::QuantizedLstm:
            return IsQuantizedLstmSupported(infos[0],
                                            infos[1],
                                            infos[2],
                                            infos[3],
                                            infos[4],
                                            quantizedLstmParamsInfo.value(),
                                            reasonIfUnsupported);
        case LayerType::Reshape:
            return IsReshapeSupported(infos[0],
                                      infos[1],
                                      *(PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::Rank:
            return IsRankSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Resize:
            return IsResizeSupported(infos[0],
                                     infos[1],
                                     *(PolymorphicDowncast<const ResizeDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        case LayerType::Reduce:
            return IsReduceSupported(infos[0],
                                     infos[1],
                                     *(PolymorphicDowncast<const ReduceDescriptor*>(&descriptor)),
                                     reasonIfUnsupported);
        case LayerType::Slice:
            return IsSliceSupported(infos[0],
                                    infos[1],
                                    *(PolymorphicDowncast<const SliceDescriptor*>(&descriptor)),
                                    reasonIfUnsupported);
        case LayerType::Softmax:
            return IsSoftmaxSupported(infos[0],
                                      infos[1],
                                      *(PolymorphicDowncast<const SoftmaxDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::SpaceToBatchNd:
            return IsSpaceToBatchNdSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const SpaceToBatchNdDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::SpaceToDepth:
            return IsSpaceToDepthSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const SpaceToDepthDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::Splitter:
        {
            std::vector<TensorInfo> outputInfos;
            for (uint32_t i = 1; i < infos.size(); i++)
            {
                outputInfos.push_back(infos[i]);
            }
            return IsSplitterSupported(infos[0],
                                       {outputInfos.begin(), outputInfos.end()},
                                       *(PolymorphicDowncast<const ViewsDescriptor*>(&descriptor)),
                                       reasonIfUnsupported);
        }
        case LayerType::Stack:
        {
            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < infos.size() - 1; i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            return IsStackSupported(inputInfos,
                                    infos[infos.size() - 1],
                                    *(PolymorphicDowncast<const StackDescriptor*>(&descriptor)),
                                    reasonIfUnsupported);
        }
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
        case LayerType::StridedSlice:
            return IsStridedSliceSupported(infos[0],
                                           infos[1],
                                           *(PolymorphicDowncast<const StridedSliceDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        case LayerType::Subtraction:
            return IsSubtractionSupported(infos[0],
                                          infos[1],
                                          infos[2],
                                          reasonIfUnsupported);
        case LayerType::Switch:
            return IsSwitchSupported(infos[0],
                                     infos[1],
                                     infos[2],
                                     infos[3],
                                     reasonIfUnsupported);
        case LayerType::Transpose:
            return IsTransposeSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const TransposeDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::TransposeConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of TransposeConvolution2d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const TransposeConvolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsTransposeConvolution2dSupported(infos[0],
                                                         infos[1],
                                                         desc,
                                                         infos[2],
                                                         EmptyOptional(),
                                                         reasonIfUnsupported);
            }
            else
            {
                return IsTransposeConvolution2dSupported(infos[0],
                                                         infos[1],
                                                         desc,
                                                         infos[2],
                                                         infos[3],
                                                         reasonIfUnsupported);
            }
        }
        case LayerType::Unmap:
            return true;
        case LayerType::Cast:
            return IsCastSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Shape:
            return IsShapeSupported(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::UnidirectionalSequenceLstm:
        {
            if (infos.size() != 6)
            {
                throw InvalidArgumentException("Invalid number of TransposeConvolution2d TensorInfos. TensorInfos "
                                               "should be of format: {input, outputStateIn, cellStateIn, "
                                               "hiddenStateOutputVal, cellStateOutputVal, output}");
            }
            auto desc = *(PolymorphicDowncast<const UnidirectionalSequenceLstmDescriptor*>(&descriptor));
            return IsUnidirectionalSequenceLstmSupported(infos[0],
                                                         infos[1],
                                                         infos[2],
                                                         infos[3],
                                                         infos[4],
                                                         infos[5],
                                                         desc,
                                                         lstmParamsInfo.value(),
                                                         reasonIfUnsupported);
        }
        case LayerType::ChannelShuffle:
            return IsChannelShuffleSupported(infos[0],
                                             infos[1],
                                             *(PolymorphicDowncast<const ChannelShuffleDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::Convolution3d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of Convolution3d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const Convolution3dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsConvolution3dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                EmptyOptional(),
                                                reasonIfUnsupported);
            }
            else
            {
                return IsConvolution3dSupported(infos[0],
                                                infos[1],
                                                desc,
                                                infos[2],
                                                infos[3],
                                                reasonIfUnsupported);
            }
        }
        case LayerType::Pooling3d:
            return IsPooling3dSupported(infos[0],
                                        infos[1],
                                        *(PolymorphicDowncast<const Pooling3dDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        default:
            return false;
    }
}

bool ILayerSupport::IsActivationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const ActivationDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsArgMinMaxSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const ArgMinMaxDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const TensorInfo& mean,
                                                  const TensorInfo& var,
                                                  const TensorInfo& beta,
                                                  const TensorInfo& gamma,
                                                  const BatchNormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, mean, var, beta, gamma, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const BatchToSpaceNdDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsCastSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsChannelShuffleSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const ChannelShuffleDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsComparisonSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          const ComparisonDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                      const TensorInfo& output,
                                      const OriginsDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(inputs, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsConstantSupported(const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const Convolution2dDescriptor& descriptor,
                                             const TensorInfo& weights,
                                             const Optional<TensorInfo>& biases,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, weights, biases, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsConvolution3dSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const Convolution3dDescriptor& descriptor,
                                             const TensorInfo& weights,
                                             const Optional<TensorInfo>& biases,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, weights, biases, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDebugSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const DepthToSpaceDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDepthwiseConvolutionSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const DepthwiseConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input,
                 output,
                 descriptor,
                 weights,
                 biases,
                 reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDequantizeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                                    const TensorInfo& scores,
                                                    const TensorInfo& anchors,
                                                    const TensorInfo& detectionBoxes,
                                                    const TensorInfo& detectionClasses,
                                                    const TensorInfo& detectionScores,
                                                    const TensorInfo& numDetections,
                                                    const DetectionPostProcessDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(boxEncodings,
                 scores,
                 anchors,
                 detectionBoxes,
                 detectionClasses,
                 detectionScores,
                 numDetections,
                 descriptor,
                 reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDilatedDepthwiseConvolutionSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const DepthwiseConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, weights, biases, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsElementwiseUnarySupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const ElementwiseUnaryDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                const FakeQuantizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsFillSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const FillDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsFloorSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const TensorInfo& weights,
                                              const TensorInfo& biases,
                                              const FullyConnectedDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, weights, biases, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsGatherSupported(const TensorInfo& input0,
                                      const TensorInfo& input1,
                                      const TensorInfo& output,
                                      const GatherDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsInputSupported(const TensorInfo& input,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsInstanceNormalizationSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const InstanceNormalizationDescriptor& descriptor,
        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const L2NormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsLogicalBinarySupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             const LogicalBinaryDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsLogicalUnarySupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ElementwiseUnaryDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const LogSoftmaxDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsLstmSupported(const TensorInfo& input,
                                    const TensorInfo& outputStateIn,
                                    const TensorInfo& cellStateIn,
                                    const TensorInfo& scratchBuffer,
                                    const TensorInfo& outputStateOut,
                                    const TensorInfo& cellStateOut,
                                    const TensorInfo& output,
                                    const LstmDescriptor& descriptor,
                                    const LstmInputParamsInfo& paramsInfo,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input,
                 outputStateIn,
                 cellStateIn,
                 scratchBuffer,
                 outputStateOut,
                 cellStateOut,
                 output,
                 descriptor,
                 paramsInfo,
                 reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMeanSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const MeanDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMemCopySupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMemImportSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMergeSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const NormalizationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsOutputSupported(const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsPadSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const PadDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsPermuteSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const PermuteDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const Pooling2dDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsPooling3dSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const Pooling3dDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsPreCompiledSupported(const TensorInfo& input,
                                           const PreCompiledDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsPreluSupported(const TensorInfo& input,
                                     const TensorInfo& alpha,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, alpha, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsQuantizeSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsQLstmSupported(const TensorInfo& input,
                                     const TensorInfo& previousOutputIn,
                                     const TensorInfo& previousCellStateIn,
                                     const TensorInfo& outputStateOut,
                                     const TensorInfo& cellStateOut,
                                     const TensorInfo& output,
                                     const QLstmDescriptor& descriptor,
                                     const LstmInputParamsInfo& paramsInfo,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input,
                 previousOutputIn,
                 previousCellStateIn,
                 outputStateOut,
                 cellStateOut,
                 output,
                 descriptor,
                 paramsInfo,
                 reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsQuantizedLstmSupported(const TensorInfo& input,
                                             const TensorInfo& previousCellStateIn,
                                             const TensorInfo& previousOutputIn,
                                             const TensorInfo& cellStateOut,
                                             const TensorInfo& output,
                                             const QuantizedLstmInputParamsInfo& paramsInfo,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input,
                 previousCellStateIn,
                 previousOutputIn,
                 cellStateOut,
                 output,
                 paramsInfo,
                 reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsRankSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsReduceSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const ReduceDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsReshapeSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const ReshapeDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsResizeSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const ResizeDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsShapeSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSliceSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const SliceDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const SoftmaxDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SpaceToBatchNdDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SpaceToDepthDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSplitterSupported(const TensorInfo& input,
                                        const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                        const ViewsDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, outputs, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                     const TensorInfo& output,
                                     const StackDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(inputs, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                                       const std::vector<const TensorInfo*>& outputs,
                                       const StandInDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(inputs, outputs, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const StridedSliceDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsSwitchSupported(const TensorInfo& input0,
                                      const TensorInfo& input1,
                                      const TensorInfo& output0,
                                      const TensorInfo& output1,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input0, input1, output0, output1, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsTransposeConvolution2dSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const TransposeConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, weights, biases, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsTransposeSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const TransposeDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input, output, descriptor, reasonIfUnsupported);
    return false;
}

bool ILayerSupport::IsUnidirectionalSequenceLstmSupported(
        const TensorInfo& input,
        const TensorInfo& outputStateIn,
        const TensorInfo& cellStateIn,
        const TensorInfo& outputStateOut,
        const TensorInfo& cellStateOut,
        const TensorInfo& output,
        const LstmDescriptor& descriptor,
        const LstmInputParamsInfo& paramsInfo,
        Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input,
                 outputStateIn,
                 cellStateIn,
                 outputStateOut,
                 cellStateOut,
                 output,
                 descriptor,
                 paramsInfo,
                 reasonIfUnsupported);
    return false;
}
ARMNN_NO_DEPRECATE_WARN_END
}
