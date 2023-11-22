//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaTensorHandle.hpp"
#include "GpuFsaTensorHandleFactory.hpp"

namespace armnn
{

using FactoryId = ITensorHandleFactory::FactoryId;

std::unique_ptr<ITensorHandle> GpuFsaTensorHandleFactory::CreateSubTensorHandle(ITensorHandle& parent,
                                                                            const TensorShape& subTensorShape,
                                                                            const unsigned int* subTensorOrigin) const
{
    arm_compute::Coordinates coords;
    arm_compute::TensorShape shape = armcomputetensorutils::BuildArmComputeTensorShape(subTensorShape);

    coords.set_num_dimensions(subTensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < subTensorShape.GetNumDimensions(); ++i)
    {
        // Arm compute indexes tensor coords in reverse order.
        unsigned int revertedIndex = subTensorShape.GetNumDimensions() - i - 1;
        coords.set(i, armnn::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    const arm_compute::TensorShape parentShape = armcomputetensorutils::BuildArmComputeTensorShape(parent.GetShape());

    // In order for ACL to support subtensors the concat axis cannot be on x or y and the values of x and y
    // must match the parent shapes
    if (coords.x() != 0 || coords.y() != 0)
    {
        return nullptr;
    }
    if ((parentShape.x() != shape.x()) || (parentShape.y() != shape.y()))
    {
        return nullptr;
    }

    if (!::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, parentShape, coords, shape))
    {
        return nullptr;
    }

    return std::make_unique<GpuFsaSubTensorHandle>(PolymorphicDowncast<IClTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<ITensorHandle> GpuFsaTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return GpuFsaTensorHandleFactory::CreateTensorHandle(tensorInfo, true);
}

std::unique_ptr<ITensorHandle> GpuFsaTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                             DataLayout dataLayout) const
{
    return GpuFsaTensorHandleFactory::CreateTensorHandle(tensorInfo, dataLayout, true);
}

std::unique_ptr<ITensorHandle> GpuFsaTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                             const bool IsMemoryManaged) const
{
    std::unique_ptr<GpuFsaTensorHandle> tensorHandle = std::make_unique<GpuFsaTensorHandle>(tensorInfo);
    if (!IsMemoryManaged)
    {
        ARMNN_LOG(warning) << "GpuFsaTensorHandleFactory only has support for memory managed.";
    }
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());
    return tensorHandle;
}

std::unique_ptr<ITensorHandle> GpuFsaTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                             DataLayout dataLayout,
                                                                             const bool IsMemoryManaged) const
{
    std::unique_ptr<GpuFsaTensorHandle> tensorHandle = std::make_unique<GpuFsaTensorHandle>(tensorInfo, dataLayout);
    if (!IsMemoryManaged)
    {
        ARMNN_LOG(warning) << "GpuFsaTensorHandleFactory only has support for memory managed.";
    }
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());
    return tensorHandle;
}

const FactoryId& GpuFsaTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(GpuFsaTensorHandleFactoryId());
    return s_Id;
}

const FactoryId& GpuFsaTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool GpuFsaTensorHandleFactory::SupportsSubTensors() const
{
    return true;
}

MemorySourceFlags GpuFsaTensorHandleFactory::GetExportFlags() const
{
    return MemorySourceFlags(MemorySource::Undefined);
}

MemorySourceFlags GpuFsaTensorHandleFactory::GetImportFlags() const
{
    return MemorySourceFlags(MemorySource::Undefined);
}

} // namespace armnn