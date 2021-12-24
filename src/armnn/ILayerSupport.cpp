//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/backends/ILayerSupport.hpp>

namespace armnn
{

ARMNN_NO_DEPRECATE_WARN_BEGIN
// IsLayerSupport() forwards to the deprecated virtual methods depending on input LayerType.
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
        case LayerType::ConvertBf16ToFp32:
            return IsConvertBf16ToFp32Supported(infos[0],
                                                infos[1],
                                                reasonIfUnsupported);
        case LayerType::ConvertFp16ToFp32:
            return IsConvertFp16ToFp32Supported(infos[0],
                                                infos[1],
                                                reasonIfUnsupported);
        case LayerType::ConvertFp32ToBf16:
            return IsConvertFp32ToBf16Supported(infos[0],
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

            bool isHiddenStateOutputOptional = (infos[4] == TensorInfo());
            bool isCellStateOutput = (infos[5] == TensorInfo());
            if (isHiddenStateOutputOptional && isCellStateOutput)
            {
                return IsUnidirectionalSequenceLstmSupported(infos[0],
                                                             infos[1],
                                                             infos[2],
                                                             infos[3],
                                                             EmptyOptional(),
                                                             EmptyOptional(),
                                                             desc,
                                                             lstmParamsInfo.value(),
                                                             reasonIfUnsupported);
            }
            else if (isHiddenStateOutputOptional)
            {
                return IsUnidirectionalSequenceLstmSupported(infos[0],
                                                             infos[1],
                                                             infos[2],
                                                             infos[3],
                                                             EmptyOptional(),
                                                             infos[5],
                                                             desc,
                                                             lstmParamsInfo.value(),
                                                             reasonIfUnsupported);
            }
            else if (isCellStateOutput)
            {
                return IsUnidirectionalSequenceLstmSupported(infos[0],
                                                             infos[1],
                                                             infos[2],
                                                             infos[3],
                                                             infos[4],
                                                             EmptyOptional(),
                                                             desc,
                                                             lstmParamsInfo.value(),
                                                             reasonIfUnsupported);
            }
            else
            {
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
ARMNN_NO_DEPRECATE_WARN_END
}
