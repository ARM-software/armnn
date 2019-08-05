//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

#if defined(VALID_TEST_DYNAMIC_BACKEND_1) || \
    defined(VALID_TEST_DYNAMIC_BACKEND_2) || \
    defined(VALID_TEST_DYNAMIC_BACKEND_3) || \
    defined(VALID_TEST_DYNAMIC_BACKEND_4) || \
    defined(VALID_TEST_DYNAMIC_BACKEND_5)

// Correct dynamic backend interface
extern "C"
{
const char* GetBackendId();
void GetVersion(uint32_t* outMajor, uint32_t* outMinor);
void* BackendFactory();
}

#elif defined(INVALID_TEST_DYNAMIC_BACKEND_1)

// Wrong external linkage: expected C-style name mangling
extern const char* GetBackendId();
extern void GetVersion(uint32_t* outMajor, uint32_t* outMinor);
extern void* BackendFactory();

#else

extern "C"
{

#if defined(INVALID_TEST_DYNAMIC_BACKEND_2)

// Invalid interface: missing "GetBackendId()"
void GetVersion(uint32_t* outMajor, uint32_t* outMinor);
void* BackendFactory();

#elif defined(INVALID_TEST_DYNAMIC_BACKEND_3)

// Invalid interface: missing "GetVersion()"
const char* GetBackendId();
void* BackendFactory();

#elif defined(INVALID_TEST_DYNAMIC_BACKEND_4)

// Invalid interface: missing "BackendFactory()"
const char* GetBackendId();
void GetVersion(uint32_t* outMajor, uint32_t* outMinor);

#elif defined(INVALID_TEST_DYNAMIC_BACKEND_5)  || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_6)  || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_7)  || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_8)  || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_9)  || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_10) || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_11)

// The interface is correct, the corresponding invalid changes are in the TestDynamicBackend.cpp file
const char* GetBackendId();
void GetVersion(uint32_t* outMajor, uint32_t* outMinor);
void* BackendFactory();

#else

#error "Invalid or missing configuration macro for the TestDynamicBackend object"

#endif

}

#endif
