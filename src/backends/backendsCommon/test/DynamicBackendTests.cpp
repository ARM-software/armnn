//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DynamicBackendTests.hpp"

#include <test/UnitTests.hpp>

BOOST_AUTO_TEST_SUITE(DynamicBackendTests)

ARMNN_SIMPLE_TEST_CASE(OpenCloseHandle, OpenCloseHandleTestImpl);
ARMNN_SIMPLE_TEST_CASE(CloseInvalidHandle, CloseInvalidHandleTestImpl);
ARMNN_SIMPLE_TEST_CASE(OpenEmptyFileName, OpenEmptyFileNameTestImpl);
ARMNN_SIMPLE_TEST_CASE(OpenNotExistingFile, OpenNotExistingFileTestImpl);
ARMNN_SIMPLE_TEST_CASE(OpenNotSharedObjectFile, OpenNotSharedObjectTestImpl);
ARMNN_SIMPLE_TEST_CASE(GetValidEntryPoint, GetValidEntryPointTestImpl);
ARMNN_SIMPLE_TEST_CASE(GetNameMangledEntryPoint, GetNameMangledEntryPointTestImpl);
ARMNN_SIMPLE_TEST_CASE(GetNoExternEntryPoint, GetNoExternEntryPointTestImpl);
ARMNN_SIMPLE_TEST_CASE(GetNotExistingEntryPoint, GetNotExistingEntryPointTestImpl);

ARMNN_SIMPLE_TEST_CASE(BackendVersioning, BackendVersioningTestImpl);

ARMNN_SIMPLE_TEST_CASE(CreateValidDynamicBackendObject, CreateValidDynamicBackendObjectTestImpl);

ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidHandle,
                       CreateDynamicBackendObjectInvalidHandleTestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface1,
                       CreateDynamicBackendObjectInvalidInterface1TestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface2,
                       CreateDynamicBackendObjectInvalidInterface2TestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface3,
                       CreateDynamicBackendObjectInvalidInterface3TestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface4,
                       CreateDynamicBackendObjectInvalidInterface4TestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface5,
                       CreateDynamicBackendObjectInvalidInterface5TestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface6,
                       CreateDynamicBackendObjectInvalidInterface6TestImpl);
ARMNN_SIMPLE_TEST_CASE(CreateDynamicBackendObjectInvalidInterface7,
                       CreateDynamicBackendObjectInvalidInterface7TestImpl);

ARMNN_SIMPLE_TEST_CASE(GetBackendPaths, GetBackendPathsTestImpl)
ARMNN_SIMPLE_TEST_CASE(GetBackendPathsOverride, GetBackendPathsOverrideTestImpl)

ARMNN_SIMPLE_TEST_CASE(GetSharedObjects, GetSharedObjectsTestImpl);

BOOST_AUTO_TEST_SUITE_END()
