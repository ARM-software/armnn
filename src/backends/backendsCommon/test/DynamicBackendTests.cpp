//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DynamicBackendTests.hpp"

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

BOOST_AUTO_TEST_SUITE_END()
