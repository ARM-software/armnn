//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if !defined(__MACH__)
.section .rodata
#define EXTERN_ASM
#else
.const_data
#define EXTERN_ASM _
#endif

#define GLUE(a, b) a ## b
#define JOIN(a, b) GLUE(a, b)
#define X(s) JOIN(EXTERN_ASM, s)

.global X(deserialize_schema_start)
.global X(deserialize_schema_end)

X(deserialize_schema_start):
.incbin ARMNN_SERIALIZER_SCHEMA_PATH
X(deserialize_schema_end):
