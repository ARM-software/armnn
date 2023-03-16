//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>

#include <Version.hpp>

#include "Activation.hpp"
#include "ArgMinMax.hpp"
#include "BatchMatMul.hpp"
#include "BatchSpace.hpp"
#include "Comparison.hpp"
#include "Convolution.hpp"
#include "Control.hpp"
#include "ElementwiseBinary.hpp"
#include "ElementwiseUnary.hpp"
#include "Fill.hpp"
#include "FullyConnected.hpp"
#include "Gather.hpp"
#include "GatherNd.hpp"
#include "LogicalBinary.hpp"
#include "Lstm.hpp"
#include "Normalization.hpp"
#include "Pack.hpp"
#include "Pad.hpp"
#include "Pooling.hpp"
#include "Prelu.hpp"
#include "Quantization.hpp"
#include "Redefine.hpp"
#include "Reduce.hpp"
#include "Resize.hpp"
#include "Round.hpp"
#include "Shape.hpp"
#include "Slice.hpp"
#include "StridedSlice.hpp"
#include "Softmax.hpp"
#include "SpaceDepth.hpp"
#include "Split.hpp"
#include "Transpose.hpp"
#include "UnidirectionalSequenceLstm.hpp"
#include "Unpack.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnnUtils/Filesystem.hpp>
#include <armnn/utility/Timer.hpp>
#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/context_util.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/logger.h>

#include <algorithm>
#include <iostream>
#include <sstream>

namespace armnnOpaqueDelegate
{

ArmnnOpaqueDelegate::ArmnnOpaqueDelegate(armnnDelegate::DelegateOptions options)
    : m_Options(std::move(options))
{
    // Configures logging for ARMNN
    if (m_Options.IsLoggingEnabled())
    {
        armnn::ConfigureLogging(true, true, m_Options.GetLoggingSeverity());
    }
    // Create/Get the static ArmNN Runtime. Note that the m_Runtime will be shared by all armnn_delegate
    // instances so the RuntimeOptions cannot be altered for different armnn_delegate instances.
    m_Runtime = GetRuntime(m_Options.GetRuntimeOptions());
    std::vector<armnn::BackendId> backends;
    if (m_Runtime)
    {
        const armnn::BackendIdSet supportedDevices = m_Runtime->GetDeviceSpec().GetSupportedBackends();
        for (auto& backend : m_Options.GetBackends())
        {
            if (std::find(supportedDevices.cbegin(), supportedDevices.cend(), backend) == supportedDevices.cend())
            {
                TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                                "TfLiteArmnnDelegate: Requested unknown backend %s", backend.Get().c_str());
            }
            else
            {
                backends.push_back(backend);
            }
        }
    }

    if (backends.empty())
    {
        // No known backend specified
        throw armnn::InvalidArgumentException("TfLiteArmnnOpaqueDelegate: No known backend specified.");
    }
    m_Options.SetBackends(backends);

    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, "TfLiteArmnnOpaqueDelegate: Created TfLite ArmNN delegate.");
}

TfLiteOpaqueDelegate* TfLiteArmnnOpaqueDelegateCreate(const void* settings)
{
    // This method will always create Opaque Delegate with default settings until
    // we have a DelegateOptions Constructor which can parse the void* settings
    armnn::IgnoreUnused(settings);
    auto options = TfLiteArmnnDelegateOptionsDefault();
    auto* armnnDelegate = new ::armnnOpaqueDelegate::ArmnnOpaqueDelegate(options);
    return TfLiteOpaqueDelegateCreate(armnnDelegate->GetDelegateBuilder());
}

::armnnDelegate::DelegateOptions TfLiteArmnnDelegateOptionsDefault()
{
    ::armnnDelegate::DelegateOptions options(armnn::Compute::CpuRef);
    return options;
}

void TfLiteArmnnOpaqueDelegateDelete(TfLiteOpaqueDelegate* tfLiteDelegate)
{
    if (tfLiteDelegate != nullptr)
    {
        delete static_cast<::armnnOpaqueDelegate::ArmnnOpaqueDelegate*>(TfLiteOpaqueDelegateGetData(tfLiteDelegate));
        TfLiteOpaqueDelegateDelete(tfLiteDelegate);
    }
}

const TfLiteOpaqueDelegatePlugin* GetArmnnDelegatePluginApi()
{
    static constexpr TfLiteOpaqueDelegatePlugin armnnPlugin{
            TfLiteArmnnOpaqueDelegateCreate, TfLiteArmnnOpaqueDelegateDelete, TfLiteArmnnOpaqueDelegateErrno};
    return &armnnPlugin;
}

const std::string ArmnnOpaqueDelegate::GetVersion() {
    return OPAQUE_DELEGATE_VERSION;
}

} // armnnOpaqueDelegate namespace