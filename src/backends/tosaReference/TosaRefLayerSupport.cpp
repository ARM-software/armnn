//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefLayerSupport.hpp"

#include <tosaCommon/TosaMappings.hpp>

#include <armnn/Types.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <graph_status.h>
#include <model_runner.h>

#include <vector>

namespace armnn
{

bool TosaRefLayerSupport::IsLayerSupported(const LayerType& type,
                                           const std::vector<TensorInfo>& infos,
                                           const BaseDescriptor& descriptor,
                                           const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                           const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmInputParamsInfo,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(lstmParamsInfo);
    IgnoreUnused(quantizedLstmInputParamsInfo);
    IgnoreUnused(reasonIfUnsupported);

    std::vector<const TensorInfo*> inputInfos;
    std::vector<const TensorInfo*> outputInfos;

    switch (type)
    {
        case LayerType::Input:
        case LayerType::Output:
            return true;
        case LayerType::Addition:
        case LayerType::Multiplication:
        case LayerType::Subtraction:
            // Setup inputs and outputs
            inputInfos.push_back(&infos[0]);
            inputInfos.push_back(&infos[1]);
            outputInfos.push_back(&infos[2]);
            break;
        case LayerType::Concat:
            for (unsigned int i = 0; i < infos.size() - 1; ++i)
            {
                inputInfos.push_back(&infos[i]);
            }
            outputInfos.push_back(&infos.back());
            break;
        case LayerType::Constant:
            outputInfos.push_back(&infos[0]);
            break;
        case LayerType::Convolution2d:
        {
            inputInfos.push_back(&infos[0]); // input
            outputInfos.push_back(&infos[1]); // output
            inputInfos.push_back(&infos[2]); // weights

            auto conv2dDesc = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor);
            if(conv2dDesc->m_BiasEnabled)
            {
                inputInfos.push_back(&infos[3]); // bias
            }
            break;
        }
        case LayerType::ElementwiseUnary:
        case LayerType::Pooling2d:
        case LayerType::Reshape:
        case LayerType::Slice:
        case LayerType::Transpose:
            inputInfos.push_back(&infos[0]);
            outputInfos.push_back(&infos[1]);
            break;
        case LayerType::TransposeConvolution2d:
        {
            inputInfos.push_back(&infos[0]); // input
            outputInfos.push_back(&infos[1]); // output
            inputInfos.push_back(&infos[2]); // weights

            auto conv2dDesc = PolymorphicDowncast<const TransposeConvolution2dDescriptor*>(&descriptor);
            if(conv2dDesc->m_BiasEnabled)
            {
                inputInfos.push_back(&infos[3]); // bias
            }
            break;
        }
        default:
            // Default to false for all unsupported layers.
            return false;
    }

    auto mappings = GetTosaMapping(nullptr, type, inputInfos, outputInfos, descriptor);
    if (mappings->GetName() == "")
    {
        // There currently isn't a TOSA mapping for this layer, as the default was returned.
        return false;
    }

    TosaSerializationHandler handler;

    // Add mappings to main block as the TOSA Reference Model requires the graph to be in one block called main.
    auto* block = new TosaSerializationBasicBlock("main",
                                                  mappings->GetOperators(),
                                                  mappings->GetTensors(),
                                                  mappings->GetInputs(),
                                                  mappings->GetOutputs());
    handler.GetBlocks().emplace_back(block);

    GraphStatus status;
    TosaReference::IModelRunner runner;

#if !defined(TOSA_REFERENCE_MODEL_OUTPUT)
    // There currently isn't a way to disable the output from the TOSA Reference Model, but it does have a file pointer
    // to write debug output to, so set this to /dev/null (if it exists on the system) to hide the output.
    func_debug_t funcDebug;

    FILE* file = fopen("/dev/null", "w");
    funcDebug.func_debug_file = (file == nullptr) ? stderr : file;

    runner.setFuncDebug(funcDebug);
#endif

    // Initialise the model runner with the TosaSerializationHandler, which runs validation on the mapping.
    status = runner.initialize(handler);

#if !defined(TOSA_REFERENCE_MODEL_OUTPUT)
    // Reset FuncDebug as they can persist across multiple IModelRunner instances.
    funcDebug.func_debug_file = stderr;
    runner.setFuncDebug(funcDebug);
#endif

    if(status == GraphStatus::TOSA_ERROR || status == GraphStatus::TOSA_UNPREDICTABLE)
    {
        return false;
    }
    else
    {
        return true;
    }
}

} // namespace armnn
