// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

///////////////////////////////////////////////////////////////////////////
//
// This API is provided as a simple interface for Microsoft SEAL library
// that can be PInvoked by .Net code.
//
///////////////////////////////////////////////////////////////////////////

#include "sealnet/defines.h"
#include <stdint.h>

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Create1(void **kswitch_keys);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Create2(void *copy, void **kswitch_keys);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Destroy(void *thisptr);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Set(void *thisptr, void *assign);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Size(void *thisptr, uint64_t *size);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_RawSize(void *thisptr, uint64_t *key_count);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_GetKeyList(void *thisptr, uint64_t index, uint64_t *count, void **key_list);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_ClearDataAndReserve(void *thisptr, uint64_t size);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_AddKeyList(void *thisptr, uint64_t count, void **key_list);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_GetParmsId(void *thisptr, uint64_t *parms_id);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_SetParmsId(void *thisptr, uint64_t *parms_id);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Pool(void *thisptr, void **pool);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_SaveSize(void *thisptr, uint8_t compr_mode, int64_t *result);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Save(void *thisptr, uint8_t *outptr, uint64_t size, uint8_t compr_mode, int64_t *out_bytes);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_UnsafeLoad(void *thisptr, void *context, uint8_t *inptr, uint64_t size, int64_t *in_bytes);

SEALNETNATIVE HRESULT SEALCALL KSwitchKeys_Load(void *thisptr, void *context, uint8_t *inptr, uint64_t size, int64_t *in_bytes);
