//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDynamicBackend.hpp"

#include <reference/RefBackend.hpp>

using namespace armnn;

const char* GetBackendId()
{
    return RefBackend::GetIdStatic().Get().c_str();
}

void GetVersion(uint32_t* outMajor, uint32_t* outMinor)
{
    if (!outMajor || !outMinor)
    {
        return;
    }

    BackendVersion apiVersion = IBackendInternal::GetApiVersion();

    *outMajor = apiVersion.m_Major;
    *outMinor = apiVersion.m_Minor;
}

void* BackendFactory()
{
    return new RefBackend();
}
