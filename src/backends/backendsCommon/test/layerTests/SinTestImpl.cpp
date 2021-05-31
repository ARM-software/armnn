//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SinTestImpl.hpp"
#include "ElementwiseUnaryTestImpl.hpp"

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Sin2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 2, 2 };

    std::vector<float> inputValues
    {
        3.0f, 2.0f,
        1.0f, 1.1f
    };

    std::vector<float> expectedOutputValues
    {
        0.14112000806f, 0.90929742682f,
        0.8414709848f, 0.89120736006f
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Sin,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Sin3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 3, 1, 2 };

    std::vector<float> inputValues
    {
        5.0f, 4.0f,
        3.0f, 2.0f,
        1.0f, 1.1f
    };

    std::vector<float> expectedOutputValues
    {
        -0.95892427466f, -0.7568024953f,
        0.14112000806f, 0.90929742682f,
        0.8414709848f, 0.89120736006f
    };

    return ElementwiseUnaryTestHelper<3, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Sin,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> SinZeroTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2 };

    std::vector<float> inputValues
    {
        0.f, 0.f
    };

    std::vector<float> expectedOutputValues
    {
        0.f, 0.f
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Sin,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> SinNegativeTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2 };

    std::vector<float> inputValues
    {
        -5.9f, -5.8f
    };

    std::vector<float> expectedOutputValues
    {
        0.37387666483f, 0.46460217941f,
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Sin,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

//
// Sinlicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
Sin2dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
Sin2dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
Sin2dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
Sin2dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Sin2dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
Sin3dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
Sin3dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
Sin3dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
Sin3dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Sin3dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
SinZeroTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
SinNegativeTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);