//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/utility/Assert.hpp>

#include <cl/ClImportTensorHandleFactory.hpp>

#include <doctest/doctest.h>

TEST_SUITE("ClImportTensorHandleFactoryTests")
{
using namespace armnn;

TEST_CASE("ImportTensorFactoryAskedToCreateManagedTensorThrowsException")
{
    // Create the factory to import tensors.
    ClImportTensorHandleFactory factory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                        static_cast<MemorySourceFlags>(MemorySource::Malloc));
    TensorInfo tensorInfo;
    // This factory is designed to import the memory of tensors. Asking for a handle that requires
    // a memory manager should result in an exception.
    REQUIRE_THROWS_AS(factory.CreateTensorHandle(tensorInfo, true), InvalidArgumentException);
    REQUIRE_THROWS_AS(factory.CreateTensorHandle(tensorInfo, DataLayout::NCHW, true), InvalidArgumentException);
}

TEST_CASE("ImportTensorFactoryCreateMallocTensorHandle")
{
    // Create the factory to import tensors.
    ClImportTensorHandleFactory factory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                        static_cast<MemorySourceFlags>(MemorySource::Malloc));
    TensorShape tensorShape{ 6, 7, 8, 9 };
    TensorInfo tensorInfo(tensorShape, armnn::DataType::Float32);
    // Start with the TensorInfo factory method. Create an import tensor handle and verify the data is
    // passed through correctly.
    auto tensorHandle = factory.CreateTensorHandle(tensorInfo);
    ARMNN_ASSERT(tensorHandle);
    ARMNN_ASSERT(tensorHandle->GetImportFlags() == static_cast<MemorySourceFlags>(MemorySource::Malloc));
    ARMNN_ASSERT(tensorHandle->GetShape() == tensorShape);

    // Same method but explicitly specifying isManaged = false.
    tensorHandle = factory.CreateTensorHandle(tensorInfo, false);
    CHECK(tensorHandle);
    ARMNN_ASSERT(tensorHandle->GetImportFlags() == static_cast<MemorySourceFlags>(MemorySource::Malloc));
    ARMNN_ASSERT(tensorHandle->GetShape() == tensorShape);

    // Now try TensorInfo and DataLayout factory method.
    tensorHandle = factory.CreateTensorHandle(tensorInfo, DataLayout::NHWC);
    CHECK(tensorHandle);
    ARMNN_ASSERT(tensorHandle->GetImportFlags() == static_cast<MemorySourceFlags>(MemorySource::Malloc));
    ARMNN_ASSERT(tensorHandle->GetShape() == tensorShape);
}

TEST_CASE("CreateSubtensorOfImportTensor")
{
    // Create the factory to import tensors.
    ClImportTensorHandleFactory factory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                        static_cast<MemorySourceFlags>(MemorySource::Malloc));
    // Create a standard inport tensor.
    TensorShape tensorShape{ 224, 224, 1, 1 };
    TensorInfo tensorInfo(tensorShape, armnn::DataType::Float32);
    auto tensorHandle = factory.CreateTensorHandle(tensorInfo);
    // Use the factory to create a 16x16 sub tensor.
    TensorShape subTensorShape{ 16, 16, 1, 1 };
    // Starting at an offset of 1x1.
    uint32_t origin[4] = { 1, 1, 0, 0 };
    auto subTensor     = factory.CreateSubTensorHandle(*tensorHandle, subTensorShape, origin);
    CHECK(subTensor);
    ARMNN_ASSERT(subTensor->GetShape() == subTensorShape);
    ARMNN_ASSERT(subTensor->GetParent() == tensorHandle.get());
}

TEST_CASE("CreateSubtensorNonZeroXYIsInvalid")
{
    // Create the factory to import tensors.
    ClImportTensorHandleFactory factory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                        static_cast<MemorySourceFlags>(MemorySource::Malloc));
    // Create a standard import tensor.
    TensorShape tensorShape{ 224, 224, 1, 1 };
    TensorInfo tensorInfo(tensorShape, armnn::DataType::Float32);
    auto tensorHandle = factory.CreateTensorHandle(tensorInfo);
    // Use the factory to create a 16x16 sub tensor.
    TensorShape subTensorShape{ 16, 16, 1, 1 };
    // This looks a bit backwards because of how Cl specifies tensors. Essentially we want to trigger our
    // check "(coords.x() != 0 || coords.y() != 0)"
    uint32_t origin[4] = { 0, 0, 1, 1 };
    auto subTensor     = factory.CreateSubTensorHandle(*tensorHandle, subTensorShape, origin);
    // We expect a nullptr.
    ARMNN_ASSERT(subTensor == nullptr);
}

TEST_CASE("CreateSubtensorXYMustMatchParent")
{
    // Create the factory to import tensors.
    ClImportTensorHandleFactory factory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                        static_cast<MemorySourceFlags>(MemorySource::Malloc));
    // Create a standard import tensor.
    TensorShape tensorShape{ 224, 224, 1, 1 };
    TensorInfo tensorInfo(tensorShape, armnn::DataType::Float32);
    auto tensorHandle = factory.CreateTensorHandle(tensorInfo);
    // Use the factory to create a 16x16 sub tensor but make the CL x and y axis different.
    TensorShape subTensorShape{ 16, 16, 2, 2 };
    // We want to trigger our ((parentShape.x() != shape.x()) || (parentShape.y() != shape.y()))
    uint32_t origin[4] = { 1, 1, 0, 0 };
    auto subTensor     = factory.CreateSubTensorHandle(*tensorHandle, subTensorShape, origin);
    // We expect a nullptr.
    ARMNN_ASSERT(subTensor == nullptr);
}

TEST_CASE("CreateSubtensorMustBeSmallerThanParent")
{
    // Create the factory to import tensors.
    ClImportTensorHandleFactory factory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                        static_cast<MemorySourceFlags>(MemorySource::Malloc));
    // Create a standard import tensor.
    TensorShape tensorShape{ 224, 224, 1, 1 };
    TensorInfo tensorInfo(tensorShape, armnn::DataType::Float32);
    auto tensorHandle = factory.CreateTensorHandle(tensorInfo);
    // Ask for a subtensor that's the same size as the parent.
    TensorShape subTensorShape{ 224, 224, 1, 1 };
    uint32_t origin[4] = { 1, 1, 0, 0 };
    // This should result in a nullptr.
    auto subTensor = factory.CreateSubTensorHandle(*tensorHandle, subTensorShape, origin);
    ARMNN_ASSERT(subTensor == nullptr);
}

}
