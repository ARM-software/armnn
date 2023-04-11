//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <DelegateOptions.hpp>

#include <tensorflow/lite/c/c_api_opaque.h>
#include <tensorflow/lite/core/experimental/acceleration/configuration/c/stable_delegate.h>

namespace armnnOpaqueDelegate
{

struct DelegateData
{
    DelegateData(const std::vector<armnn::BackendId>& backends)
    : m_Backends(backends)
    , m_Network(nullptr, nullptr)
    {}

    const std::vector<armnn::BackendId>       m_Backends;
    armnn::INetworkPtr                        m_Network;
    std::vector<armnn::IOutputSlot*>          m_OutputSlotForNode;
};

/// Forward declaration for functions initializing the ArmNN Delegate
::armnnDelegate::DelegateOptions TfLiteArmnnDelegateOptionsDefault();

TfLiteOpaqueDelegate* TfLiteArmnnOpaqueDelegateCreate(const void* settings);

void TfLiteArmnnOpaqueDelegateDelete(TfLiteOpaqueDelegate* tfLiteDelegate);

TfLiteStatus DoPrepare(TfLiteOpaqueContext* context, TfLiteOpaqueDelegate* delegate, void* data);

/// ArmNN Opaque Delegate
class ArmnnOpaqueDelegate
{
    friend class ArmnnSubgraph;
public:
    explicit ArmnnOpaqueDelegate(armnnDelegate::DelegateOptions options);

    TfLiteIntArray* IdentifyOperatorsToDelegate(TfLiteOpaqueContext* context);

    TfLiteOpaqueDelegateBuilder* GetDelegateBuilder() { return &m_Builder; }

    /// Retrieve version in X.Y.Z form
    static const std::string GetVersion();

private:
    /**
     * Returns a pointer to the armnn::IRuntime* this will be shared by all armnn_delegates.
     */
    armnn::IRuntime* GetRuntime(const armnn::IRuntime::CreationOptions& options)
    {
        static armnn::IRuntimePtr instance = armnn::IRuntime::Create(options);
        /// Instantiated on first use.
        return instance.get();
    }

    TfLiteOpaqueDelegateBuilder m_Builder =
    {
            reinterpret_cast<void*>(this),  // .data_
            DoPrepare,                      // .Prepare
            nullptr,                        // .CopyFromBufferHandle
            nullptr,                        // .CopyToBufferHandle
            nullptr,                        // .FreeBufferHandle
            kTfLiteDelegateFlagsNone,       // .flags
    };

    /// ArmNN Runtime pointer
    armnn::IRuntime* m_Runtime;
    /// ArmNN Delegate Options
    armnnDelegate::DelegateOptions m_Options;
};

static int TfLiteArmnnOpaqueDelegateErrno(TfLiteOpaqueDelegate* delegate) { return 0; }

/// In order for the delegate to be loaded by TfLite
const TfLiteOpaqueDelegatePlugin* GetArmnnDelegatePluginApi();

extern const TfLiteStableDelegate TFL_TheStableDelegate;

/// ArmnnSubgraph class where parsing the nodes to ArmNN format and creating the ArmNN Graph
class ArmnnSubgraph
{
public:
    static ArmnnSubgraph* Create(TfLiteOpaqueContext* tfLiteContext,
                                 const TfLiteOpaqueDelegateParams* parameters,
                                 const ArmnnOpaqueDelegate* delegate);

    TfLiteStatus Prepare(TfLiteOpaqueContext* tfLiteContext);

    TfLiteStatus Invoke(TfLiteOpaqueContext* tfLiteContext, TfLiteOpaqueNode* tfLiteNode);

    static TfLiteStatus VisitNode(DelegateData& delegateData,
                                  TfLiteOpaqueContext* tfLiteContext,
                                  TfLiteRegistrationExternal* tfLiteRegistration,
                                  TfLiteOpaqueNode* tfLiteNode,
                                  int nodeIndex);
private:
    ArmnnSubgraph(armnn::NetworkId networkId,
                  armnn::IRuntime* runtime,
                  std::vector<armnn::BindingPointInfo>& inputBindings,
                  std::vector<armnn::BindingPointInfo>& outputBindings)
    : m_NetworkId(networkId)
    , m_Runtime(runtime)
    , m_InputBindings(inputBindings)
    , m_OutputBindings(outputBindings)
    {}
    static TfLiteStatus AddInputLayer(DelegateData& delegateData,
                                      TfLiteOpaqueContext* tfLiteContext,
                                      const TfLiteIntArray* inputs,
                                      std::vector<armnn::BindingPointInfo>& inputBindings);
    static TfLiteStatus AddOutputLayer(DelegateData& delegateData,
                                       TfLiteOpaqueContext* tfLiteContext,
                                       const TfLiteIntArray* outputs,
                                       std::vector<armnn::BindingPointInfo>& outputBindings);
    /// The Network Id
    armnn::NetworkId m_NetworkId;
    /// ArmNN Runtime
    armnn::IRuntime* m_Runtime;
    /// Binding information for inputs and outputs
    std::vector<armnn::BindingPointInfo> m_InputBindings;
    std::vector<armnn::BindingPointInfo> m_OutputBindings;
};

} // armnnOpaqueDelegate namespace