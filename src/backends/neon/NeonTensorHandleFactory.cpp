//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonTensorHandleFactory.hpp"
#include "NeonTensorHandle.hpp"

#include <boost/core/ignore_unused.hpp>

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
        coords.set(i, boost::numeric_cast<int>(subTensorOrigin[revertedIndex]));
    }

    const arm_compute::TensorShape parentShape = armcomputetensorutils::BuildArmComputeTensorShape(parent.GetShape());
    if (!::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, parentShape, coords, shape))
    {
        return nullptr;
    }

    return std::make_unique<NeonSubTensorHandle>(
            boost::polymorphic_downcast<IAclTensorHandle*>(&parent), shape, coords);
}

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

    return tensorHandle;
}

std::unique_ptr<ITensorHandle> NeonTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout) const
{
    auto tensorHandle = std::make_unique<NeonTensorHandle>(tensorInfo, dataLayout);
    tensorHandle->SetMemoryGroup(m_MemoryManager->GetInterLayerMemoryGroup());

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

} // namespace armnn
