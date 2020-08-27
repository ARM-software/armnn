//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReshapeTestImpl.hpp"
#include "ElementwiseUnaryTestImpl.hpp"


template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Rsqrt2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 2, 2 };

    std::vector<float> inputValues
    {
        1.f, 4.f,
        16.f, 25.f
    };

    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f,
        0.25f, 0.2f
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Rsqrt,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Rsqrt3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 3, 1, 2 };

    std::vector<float> inputValues
    {
        1.f, 4.f, 16.f,
        25.f, 64.f, 100.f
    };

    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f, 0.25f,
        0.2f, 0.125f, 0.1f
    };

    return ElementwiseUnaryTestHelper<3, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Rsqrt,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> RsqrtZeroTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2 };

    std::vector<float> inputValues
    {
        0.f, -0.f
    };

    std::vector<float> expectedOutputValues
    {
        INFINITY, -INFINITY
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Rsqrt,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> RsqrtNegativeTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2 };

    std::vector<float> inputValues
    {
        -25.f, -16.f
    };

    std::vector<float> expectedOutputValues
    {
        -NAN, -NAN
    };

    return ElementwiseUnaryTestHelper<2, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::UnaryOperation::Rsqrt,
        inputShape,
        inputValues,
        inputShape,
        expectedOutputValues,
        tensorHandleFactory);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
Rsqrt2dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
Rsqrt2dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
Rsqrt2dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
Rsqrt2dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Rsqrt2dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
Rsqrt3dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
Rsqrt3dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
Rsqrt3dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
Rsqrt3dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Rsqrt3dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
RsqrtZeroTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
RsqrtNegativeTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);
