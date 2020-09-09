//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonTensorHandleFactory.hpp"
#include "NeonTensorHandle.hpp"

#include "Layer.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

using FactoryId = ITensorHandleFactory::FactoryId;

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateSubTensorHandle(ITensorHandle& parent,
                                                                              const TensorShape& subTensorShape,
                                                                              const unsigned int* subTensorOrigin)
                                                                              const
{
    const arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    arm_compute::Coordinates coords;
    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); ++i)
    {
        // Arm compute indexes tensor coords in reverse order.
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, armnn::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    const arm_compute::TensorShape parentShape = armcomputetensorutils::BuildArmComputeTensorShape(parent.GetShape());

    if (!::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, parentShape, coords, shape))
    {
        return nullptr;
    }

    return std::make_unique<NeonSubTensorHandle>(
            PolymorphicDowncast<IAclTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return NeonTensorHandleFactory::CreateTensorHandle(tensorInfo, true);
}

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout) const
{
    return NeonTensorHandleFactory::CreateTensorHandle(tensorInfo, dataLayout, true);
}

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           const bool IsMemoryManaged) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo);
    if (IsMemoryManaged)
    {
        tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());
    }
    // If we are not Managing the Memory then we must be importing
    tensorHandle->SetImportEnabledFlag(!IsMemoryManaged);
    tensorHandle->SetImportFlags(GetImportFlags());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout,
                                                                           const bool IsMemoryManaged) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo, dataLayout);
    if (IsMemoryManaged)
    {
        tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());
    }
    // If we are not Managing the Memory then we must be importing
    tensorHandle->SetImportEnabledFlag(!IsMemoryManaged);
    tensorHandle->SetImportFlags(GetImportFlags());

    return tensorHandle;
}

const FactoryId& NeonTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(NeonTensorHandleFactoryId());
    return s_Id;
}

const FactoryId& NeonTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool NeonTensorHandleFactory::SupportsInPlaceComputation() const
{
    return true;
}

bool NeonTensorHandleFactory::SupportsSubTensors() const
{
    return true;
}

MemorySourceFlags NeonTensorHandleFactory::GetExportFlags() const
{
    return m_ExportFlags;
}

MemorySourceFlags NeonTensorHandleFactory::GetImportFlags() const
{
    return m_ImportFlags;
}

std::vector<Capability> NeonTensorHandleFactory::GetCapabilities(const IConnectableLayer* layer,
                                                                 const IConnectableLayer* connectedLayer,
                                                                 CapabilityClass capabilityClass)

{
    IgnoreUnused(connectedLayer);
    std::vector<Capability> capabilities;
    if (capabilityClass == CapabilityClass::PaddingRequired)
    {
        auto search = paddingRequiredLayers.find((PolymorphicDowncast<const Layer*>(layer))->GetType());
        if ( search != paddingRequiredLayers.end())
        {
            Capability paddingCapability(CapabilityClass::PaddingRequired, true);
            capabilities.push_back(paddingCapability);
        }
    }
    return capabilities;
}

} // namespace armnn
