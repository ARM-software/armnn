//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#if defined (_MSC_VER)

#ifdef ARMNN_COMPILING_DLL
#define ARMNN_DLLEXPORT __declspec(dllexport)
#else
#define ARMNN_DLLEXPORT __declspec(dllimport)
#endif

#else

#define ARMNN_DLLEXPORT

#endif