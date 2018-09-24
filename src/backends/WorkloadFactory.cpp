//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "WorkloadFactory.hpp"
#include <backends/reference/RefWorkloadFactory.hpp>
#include <backends/neon/NeonWorkloadFactory.hpp>
#include "ClWorkloadFactory.hpp"

#include <armnn/Types.hpp>
#include <armnn/LayerSupport.hpp>
#include <Layer.hpp>
#include <LayersFwd.hpp>
#include "CpuTensorHandle.hpp"

#include <boost/cast.hpp>
#include <cstring>
#include <boost/iterator/transform_iterator.hpp>

namespace armnn
{

namespace
{
    const TensorInfo OverrideDataType(const TensorInfo& info, boost::optional<DataType> type)
    {
        if (type == boost::none)
        {
            return info;
        }

        return TensorInfo(info.GetShape(), type.get(), info.GetQuantizationScale(), info.GetQuantizationOffset());
    }

    boost::optional<DataType> GetBiasTypeFromWeightsType(boost::optional<DataType> weightsType)
    {
        if (weightsType == boost::none)
        {
            return weightsType;
        }

        switch(weightsType.get())
        {
            case DataType::Float16:
            case DataType::Float32:
                return weightsType;
            case DataType::QuantisedAsymm8:
                return DataType::Signed32;
            default:
                BOOST_ASSERT_MSG(false, "GetBiasTypeFromWeightsType(): Unsupported data type.");
        }
        return boost::none;
    }
}

bool IWorkloadFactory::IsLayerSupported(Compute compute, const Layer& layer, boost::optional<DataType> dataType,
    std::string& outReasonIfUnsupported)
{
    constexpr size_t reasonCapacity = 1024;
    char reason[reasonCapacity];
    bool result;
    switch(layer.GetType())
    {
        case LayerType::Activation:
        {
            auto cLayer = boost::polymorphic_downcast<const ActivationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsActivationSupported(compute,
                                           OverrideDataType(input, dataType),
                                           OverrideDataType(output, dataType),
                                           cLayer->GetParameters(),
                                           reason,
                                           reasonCapacity);
            break;
        }
        case LayerType::Addition:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsAdditionSupported(compute,
                                        OverrideDataType(input0, dataType),
                                        OverrideDataType(input1, dataType),
                                        OverrideDataType(output, dataType),
                                        reason,
                                        reasonCapacity);
            break;
        }
        case LayerType::BatchNormalization:
        {
            auto cLayer = boost::polymorphic_downcast<const BatchNormalizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& mean = cLayer->m_Mean->GetTensorInfo();
            const TensorInfo& var = cLayer->m_Variance->GetTensorInfo();
            const TensorInfo& beta = cLayer->m_Beta->GetTensorInfo();
            const TensorInfo& gamma = cLayer->m_Gamma->GetTensorInfo();
            result = IsBatchNormalizationSupported(compute,
                                                   OverrideDataType(input, dataType),
                                                   OverrideDataType(output, dataType),
                                                   OverrideDataType(mean, dataType),
                                                   OverrideDataType(var, dataType),
                                                   OverrideDataType(beta, dataType),
                                                   OverrideDataType(gamma, dataType),
                                                   cLayer->GetParameters(),
                                                   reason, reasonCapacity);
            break;
        }
        case LayerType::Constant:
        {
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsConstantSupported(compute, OverrideDataType(output, dataType), reason, reasonCapacity);
            break;
        }
        case LayerType::ConvertFp16ToFp32:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsConvertFp16ToFp32Supported(compute, input, output, reason, reasonCapacity);
            break;
        }
        case LayerType::ConvertFp32ToFp16:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsConvertFp32ToFp16Supported(compute, input, output, reason, reasonCapacity);
            break;
        }
        case LayerType::Convolution2d:
        {
            auto cLayer = boost::polymorphic_downcast<const Convolution2dLayer*>(&layer);

            const TensorInfo input  = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo output = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            BOOST_ASSERT(cLayer->m_Weight.get() != nullptr);

            const Convolution2dDescriptor& descriptor  = cLayer->GetParameters();

            // Construct optional biases object based on the value of m_BiasEnabled
            boost::optional<TensorInfo> biases(boost::none);
            if (descriptor.m_BiasEnabled)
            {
                biases = boost::make_optional(
                    OverrideDataType(cLayer->m_Bias->GetTensorInfo(), GetBiasTypeFromWeightsType(dataType)));
            }

            result = IsConvolution2dSupported(compute,
                                              input,
                                              output,
                                              descriptor,
                                              OverrideDataType(cLayer->m_Weight->GetTensorInfo(), dataType),
                                              biases,
                                              reason,
                                              reasonCapacity);
            break;
        }
        case LayerType::MemCopy:
        {
            // MemCopy supported for CpuRef, CpuAcc and GpuAcc backends,
            // (also treat Undefined as CpuRef to avoid breaking lots of Unit tests).
            result = compute == Compute::CpuRef || compute == Compute::Undefined
                || compute == Compute::CpuAcc || compute == Compute::GpuAcc;
            strcpy(reason, "Unsupported backend type");
            break;
        }
        case LayerType::DepthwiseConvolution2d:
        {
            auto cLayer = boost::polymorphic_downcast<const DepthwiseConvolution2dLayer*>(&layer);
            const TensorInfo& input = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo& output = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            BOOST_ASSERT(cLayer->m_Weight.get() != nullptr);

            const DepthwiseConvolution2dDescriptor& descriptor = cLayer->GetParameters();

            // Construct optional biases object based on the value of m_BiasEnabled
            boost::optional<TensorInfo> biases(boost::none);
            if (descriptor.m_BiasEnabled)
            {
                biases = boost::make_optional(
                    OverrideDataType(cLayer->m_Bias->GetTensorInfo(), GetBiasTypeFromWeightsType(dataType)));
            }

            result = IsDepthwiseConvolutionSupported(compute,
                                                     input,
                                                     output,
                                                     descriptor,
                                                     OverrideDataType(cLayer->m_Weight->GetTensorInfo(), dataType),
                                                     biases,
                                                     reason,
                                                     reasonCapacity);
            break;
        }
        case LayerType::FakeQuantization:
        {
            auto cLayer = boost::polymorphic_downcast<const FakeQuantizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsFakeQuantizationSupported(compute, OverrideDataType(input, dataType), cLayer->GetParameters(),
                                                 reason, reasonCapacity);
            break;
        }
        case LayerType::Floor:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsFloorSupported(compute, OverrideDataType(input, dataType), OverrideDataType(output, dataType),
                                      reason, reasonCapacity);
            break;
        }
        case LayerType::FullyConnected:
        {
            auto cLayer = boost::polymorphic_downcast<const FullyConnectedLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            BOOST_ASSERT(cLayer->m_Weight.get() != nullptr);

            TensorInfo biasInfo;
            const TensorInfo * biasInfoPtr = nullptr;
            static const TensorInfo dummyFloat16Bias(TensorShape({1,1,1,1}), DataType::Float16);
            static const TensorInfo dummyFloat32Bias(TensorShape({1,1,1,1}), DataType::Float32);
            static const TensorInfo dummyQA8Bias(TensorShape({1,1,1,1}), DataType::Signed32);

            const FullyConnectedDescriptor& descriptor = cLayer->GetParameters();
            if (descriptor.m_BiasEnabled)
            {
                BOOST_ASSERT(cLayer->m_Bias.get() != nullptr);
                biasInfo = OverrideDataType(cLayer->m_Bias->GetTensorInfo(), GetBiasTypeFromWeightsType(dataType));
                biasInfoPtr = &biasInfo;
            }
            else
            {
                // If biases are not enabled pass a dummy tensorinfo for the validation
                switch(input.GetDataType())
                {
                    case DataType::Float16:
                    {
                        biasInfoPtr = &dummyFloat16Bias;
                        break;
                    }
                    case DataType::Float32:
                    {
                        biasInfoPtr = &dummyFloat32Bias;
                        break;
                    }
                    case DataType::QuantisedAsymm8:
                    {
                        biasInfoPtr = &dummyQA8Bias;
                        break;
                    }
                    default:
                    {
                        BOOST_ASSERT_MSG(false, "Unexpected bias type");
                    }
                }
            }

            result = IsFullyConnectedSupported(compute,
                                               OverrideDataType(input, dataType),
                                               OverrideDataType(output, dataType),
                                               OverrideDataType(cLayer->m_Weight->GetTensorInfo(), dataType),
                                               *biasInfoPtr,
                                               descriptor,
                                               reason,
                                               reasonCapacity);
            break;
        }
        case LayerType::Input:
        {
            const TensorInfo& input = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsInputSupported(compute, OverrideDataType(input, dataType), reason, reasonCapacity);
            break;
        }
        case LayerType::L2Normalization:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsL2NormalizationSupported(compute, OverrideDataType(input, dataType),
                    OverrideDataType(output, dataType), reason, reasonCapacity);
            break;
        }
        case LayerType::Lstm:
        {
            auto cLayer = boost::polymorphic_downcast<const LstmLayer*>(&layer);
            const LstmDescriptor& descriptor = cLayer->GetParameters();

            // All inputs.
            const TensorInfo& input = OverrideDataType(layer.GetInputSlot(0).GetConnection()->GetTensorInfo(),
                                                       dataType);
            const TensorInfo& outputStateIn = OverrideDataType(layer.GetInputSlot(1).GetConnection()->GetTensorInfo(),
                                                               dataType);
            const TensorInfo& cellStateIn = OverrideDataType(layer.GetInputSlot(2).GetConnection()->GetTensorInfo(),
                                                             dataType);
            // All outputs
            const TensorInfo& scratchBuffer = OverrideDataType(layer.GetOutputSlot(0).GetTensorInfo(), dataType);
            const TensorInfo& outputStateOut = OverrideDataType(layer.GetOutputSlot(1).GetTensorInfo(), dataType);
            const TensorInfo& cellStateOut = OverrideDataType(layer.GetOutputSlot(2).GetTensorInfo(), dataType);
            const TensorInfo& output = OverrideDataType(layer.GetOutputSlot(3).GetTensorInfo(), dataType);

            // Basic parameters
            const TensorInfo& inputToForgetWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToForgetWeights->GetTensorInfo(), dataType);
            const TensorInfo& inputToCellWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToCellWeights->GetTensorInfo(), dataType);
            const TensorInfo& inputToOutputWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_InputToOutputWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToForgetWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToForgetWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToCellWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToCellWeights->GetTensorInfo(), dataType);
            const TensorInfo& recurrentToOutputWeights
                    = OverrideDataType(cLayer->m_BasicParameters.m_RecurrentToOutputWeights->GetTensorInfo(), dataType);
            const TensorInfo& forgetGateBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_ForgetGateBias->GetTensorInfo(), dataType);
            const TensorInfo& cellBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_CellBias->GetTensorInfo(), dataType);
            const TensorInfo& outputGateBias
                    = OverrideDataType(cLayer->m_BasicParameters.m_OutputGateBias->GetTensorInfo(), dataType);

            // Optional parameters
            const TensorInfo* inputToInputWeights = nullptr;
            const TensorInfo* recurrentToInputWeights = nullptr;
            const TensorInfo* cellToInputWeights = nullptr;
            const TensorInfo* inputGateBias = nullptr;
            const TensorInfo* projectionWeights = nullptr;
            const TensorInfo* projectionBias = nullptr;
            const TensorInfo* cellToForgetWeights = nullptr;
            const TensorInfo* cellToOutputWeights = nullptr;

            TensorInfo optInputToInputWeights;
            TensorInfo optRecurrentToInputWeights;
            TensorInfo optCellToInputWeights;
            TensorInfo optInputGateBias;
            TensorInfo optProjectionWeights;
            TensorInfo optProjectionBias;
            TensorInfo optCellToForgetWeights;
            TensorInfo optCellToOutputWeights;

            if(!descriptor.m_CifgEnabled)
            {
                optInputToInputWeights =
                    OverrideDataType(cLayer->m_CifgParameters.m_InputToInputWeights->GetTensorInfo(), dataType);
                inputToInputWeights = &optInputToInputWeights;

                optRecurrentToInputWeights =
                    OverrideDataType(cLayer->m_CifgParameters.m_RecurrentToInputWeights->GetTensorInfo(), dataType);
                recurrentToInputWeights = &optRecurrentToInputWeights;
                if (cLayer->m_CifgParameters.m_CellToInputWeights != nullptr)
                {
                    optCellToInputWeights =
                        OverrideDataType(cLayer->m_CifgParameters.m_CellToInputWeights->GetTensorInfo(), dataType);
                    cellToInputWeights = &optCellToInputWeights;
                }
                optInputGateBias =
                       OverrideDataType(cLayer->m_CifgParameters.m_InputGateBias->GetTensorInfo(), dataType);
                inputGateBias = &optInputGateBias;
            }

            if(descriptor.m_ProjectionEnabled)
            {
                optProjectionWeights =
                    OverrideDataType(cLayer->m_ProjectionParameters.m_ProjectionWeights->GetTensorInfo(), dataType);
                projectionWeights = &optProjectionWeights;
                if (cLayer->m_ProjectionParameters.m_ProjectionBias != nullptr)
                {
                    optProjectionBias =
                        OverrideDataType(cLayer->m_ProjectionParameters.m_ProjectionBias->GetTensorInfo(), dataType);
                    projectionBias = &optProjectionBias;
                }
            }

            if(descriptor.m_PeepholeEnabled)
            {
                optCellToForgetWeights =
                    OverrideDataType(cLayer->m_PeepholeParameters.m_CellToForgetWeights->GetTensorInfo(), dataType);
                cellToForgetWeights = &optCellToForgetWeights;
                optCellToOutputWeights =
                    OverrideDataType(cLayer->m_PeepholeParameters.m_CellToOutputWeights->GetTensorInfo(), dataType);
                cellToOutputWeights = &optCellToOutputWeights;
            }

            result = IsLstmSupported(compute,
                                     input,
                                     outputStateIn,
                                     cellStateIn,
                                     scratchBuffer,
                                     outputStateOut,
                                     cellStateOut,
                                     output,
                                     descriptor,
                                     inputToForgetWeights,
                                     inputToCellWeights,
                                     inputToOutputWeights,
                                     recurrentToForgetWeights,
                                     recurrentToCellWeights,
                                     recurrentToOutputWeights,
                                     forgetGateBias,
                                     cellBias,
                                     outputGateBias,
                                     inputToInputWeights,
                                     recurrentToInputWeights,
                                     cellToInputWeights,
                                     inputGateBias,
                                     projectionWeights,
                                     projectionBias,
                                     cellToForgetWeights,
                                     cellToOutputWeights,
                                     reason,
                                     reasonCapacity);
            break;
        }
        case LayerType::Merger:
        {
            auto cLayer = boost::polymorphic_downcast<const MergerLayer*>(&layer);

            // Get vector of all inputs.
            auto getTensorInfo = [&dataType](const InputSlot& slot)
                {
                    return OverrideDataType(slot.GetConnectedOutputSlot()->GetTensorInfo(), dataType);
                };
            auto beginI = boost::make_transform_iterator(layer.GetInputSlots().begin(), getTensorInfo);
            auto endI = boost::make_transform_iterator(layer.GetInputSlots().end(), getTensorInfo);
            std::vector<TensorInfo> inputs(beginI, endI);

            auto getTensorInfoPtr = [](const TensorInfo& info)
                {
                    return &info;
                };
            auto beginPtr = boost::make_transform_iterator(inputs.begin(), getTensorInfoPtr);
            auto endPtr = boost::make_transform_iterator(inputs.end(), getTensorInfoPtr);
            std::vector<const TensorInfo*> inputPtrs(beginPtr, endPtr);

            result = IsMergerSupported(compute, inputPtrs, cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Multiplication:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsMultiplicationSupported(compute,
                                               OverrideDataType(input0, dataType),
                                               OverrideDataType(input1, dataType),
                                               OverrideDataType(output, dataType),
                                               reason,
                                               reasonCapacity);
            break;
        }
        case LayerType::Normalization:
        {
            auto cLayer = boost::polymorphic_downcast<const NormalizationLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsNormalizationSupported(compute, OverrideDataType(input, dataType),
                                              OverrideDataType(output, dataType), cLayer->GetParameters(), reason,
                                              reasonCapacity);
            break;
        }
        case LayerType::Output:
        {
            const TensorInfo& output = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsOutputSupported(compute, OverrideDataType(output, dataType), reason, reasonCapacity);
            break;
        }
        case LayerType::Permute:
        {
            auto cLayer = boost::polymorphic_downcast<const PermuteLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsPermuteSupported(compute, OverrideDataType(input, dataType), OverrideDataType(output, dataType),
                                        cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Pad:
        {
            auto cLayer = boost::polymorphic_downcast<const PadLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsPadSupported(compute,
                                    OverrideDataType(input, dataType),
                                    OverrideDataType(output, dataType),
                                    cLayer->GetParameters(),
                                    reason,
                                    reasonCapacity);
            break;
        }
        case LayerType::Pooling2d:
        {
            auto cLayer = boost::polymorphic_downcast<const Pooling2dLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsPooling2dSupported(compute, OverrideDataType(input, dataType),
                                          OverrideDataType(output, dataType), cLayer->GetParameters(), reason,
                                          reasonCapacity);
            break;
        }
        case LayerType::Division:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsDivisionSupported(compute,
                                         OverrideDataType(input0, dataType),
                                         OverrideDataType(input1, dataType),
                                         OverrideDataType(output, dataType),
                                         reason,
                                         reasonCapacity);
            break;
        }
        case LayerType::Reshape:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsReshapeSupported(compute, OverrideDataType(input, dataType), reason, reasonCapacity);
            break;
        }
        case LayerType::ResizeBilinear:
        {
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsResizeBilinearSupported(compute, OverrideDataType(input, dataType), reason, reasonCapacity);
            break;
        }
        case LayerType::Softmax:
        {
            auto cLayer = boost::polymorphic_downcast<const SoftmaxLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsSoftmaxSupported(compute, OverrideDataType(input, dataType), OverrideDataType(output, dataType),
                                        cLayer->GetParameters(), reason, reasonCapacity);
            break;
        }
        case LayerType::Splitter:
        {
            auto cLayer = boost::polymorphic_downcast<const SplitterLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            result = IsSplitterSupported(compute, OverrideDataType(input, dataType), cLayer->GetParameters(), reason,
                                         reasonCapacity);
            break;
        }
        case LayerType::Subtraction:
        {
            const TensorInfo& input0 = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& input1 = layer.GetInputSlot(1).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsSubtractionSupported(compute,
                                            OverrideDataType(input0, dataType),
                                            OverrideDataType(input1, dataType),
                                            OverrideDataType(output, dataType),
                                            reason,
                                            reasonCapacity);
            break;
        }
        case LayerType::Mean:
        {
            auto cLayer = boost::polymorphic_downcast<const MeanLayer*>(&layer);
            const TensorInfo& input = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
            const TensorInfo& output = layer.GetOutputSlot(0).GetTensorInfo();
            result = IsMeanSupported(compute,
                                     OverrideDataType(input, dataType),
                                     OverrideDataType(output, dataType),
                                     cLayer->GetParameters(),
                                     reason,
                                     reasonCapacity);
            break;
        }
        default:
        {
            BOOST_ASSERT_MSG(false, "WorkloadFactory did not recognise type of layer.");
            strcpy(reason, "Unrecognised layer type");
            result = false;
            break;
        }
    }
    outReasonIfUnsupported = reason;
    return result;
}

bool IWorkloadFactory::IsLayerSupported(const Layer& layer, boost::optional<DataType> dataType,
                                        std::string& outReasonIfUnsupported)
{
    return IsLayerSupported(layer.GetComputeDevice(), layer, dataType, outReasonIfUnsupported);
}

}
