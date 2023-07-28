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

bool LayerSupportBase::IsShapeSupported(const TensorInfo&, // input
                                        const TensorInfo&, // output
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

} // namespace armnn
