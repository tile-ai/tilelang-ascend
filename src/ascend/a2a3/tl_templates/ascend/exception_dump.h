#ifndef TILELANG_EXCEPTION_DUMP_H_
#define TILELANG_EXCEPTION_DUMP_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/acl_dump.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TILE_LANG_PARAM_INFO_MAGIC 0x474e414c454c4954ULL
#define TILE_LANG_MAX_TENSOR_COUNT 16
#define TILE_LANG_MAX_KERNEL_NAME_LEN 128

struct ParamSizeInfo {
  uint64_t magic;
  size_t count;
  char kernel_name[TILE_LANG_MAX_KERNEL_NAME_LEN];
  size_t sizes[TILE_LANG_MAX_TENSOR_COUNT];
  uint64_t addr[TILE_LANG_MAX_TENSOR_COUNT];
  int32_t dataTypes[TILE_LANG_MAX_TENSOR_COUNT];
};

extern "C" int tilelang_dump_from_host_args(const void *argsBuf,
                                            uint32_t argsLen) {
  if (argsBuf == nullptr || argsLen == 0) {
    return -1;
  }

  const uint8_t *hostArgs = (const uint8_t *)argsBuf;

  ParamSizeInfo *paramInfo = nullptr;
  uint32_t searchLen = argsLen / 8;
  for (uint32_t i = 0; i < searchLen; i++) {
    if (((const uint64_t *)hostArgs)[i] == TILE_LANG_PARAM_INFO_MAGIC) {
      uint32_t offset = i * 8;
      if (offset + sizeof(ParamSizeInfo) <= argsLen) {
        paramInfo = (ParamSizeInfo *)(hostArgs + offset);
      }
      break;
    }
  }

  if (paramInfo == nullptr) {
    return -2;
  }

  if (paramInfo->count == 0 || paramInfo->count > TILE_LANG_MAX_TENSOR_COUNT) {
    return -3;
  }

  size_t count = paramInfo->count;
  acldumpTensorInfo *tensorInfos =
      (acldumpTensorInfo *)calloc(count, sizeof(acldumpTensorInfo));
  if (tensorInfos == nullptr) {
    return -4;
  }

  for (size_t t = 0; t < count; t++) {
    uint64_t devAddr = paramInfo->addr[t];
    tensorInfos[t].type = ACL_DUMP_TENSOR_INPUT;
    tensorInfos[t].tensorSize = paramInfo->sizes[t];
    tensorInfos[t].format = 2;
    tensorInfos[t].dataType = paramInfo->dataTypes[t];
    tensorInfos[t].tensorAddr = (int64_t *)devAddr;
    tensorInfos[t].argsOffset = 0;
    tensorInfos[t].shape[0] = paramInfo->sizes[t];
    tensorInfos[t].originShape[0] = paramInfo->sizes[t];
  }

  if (acldumpSaveExceptionInfo != nullptr) {
    acldumpSaveExceptionInfo(paramInfo->kernel_name, "tilelang", tensorInfos,
                             count);
  } else {
    const char *dumpDir = getenv("TILELANG_EXCEPTION_DUMP_DIR");
    if (dumpDir == nullptr || dumpDir[0] == '\0') {
      dumpDir = "/tmp";
    }

    time_t now = time(nullptr);
    char filePath[512];
    snprintf(filePath, sizeof(filePath),
             "%s/tilelang_exception_dump_%d_%ld.log", dumpDir, (int)getpid(),
             (long)now);

    FILE *fp = fopen(filePath, "w");
    if (fp == nullptr) {
      snprintf(filePath, sizeof(filePath),
               "/tmp/tilelang_exception_dump_%d_%ld.log", (int)getpid(),
               (long)now);
      fp = fopen(filePath, "w");
    }
    if (fp != nullptr) {
      fprintf(fp, "TileLang Exception Dump\n");
      fprintf(fp, "Tensor count: %zu\n", count);
      for (size_t t = 0; t < count; t++) {
        uint64_t devAddr = (uint64_t)tensorInfos[t].tensorAddr;
        fprintf(fp, "tensor[%zu]: addr=0x%lx, size=%zu bytes, dataType=%d\n", t,
                (unsigned long)devAddr, tensorInfos[t].tensorSize,
                tensorInfos[t].dataType);
        if (devAddr != 0 && tensorInfos[t].tensorSize > 0) {
          size_t dumpBytes = tensorInfos[t].tensorSize;
          if (dumpBytes > 128)
            dumpBytes = 128;
          uint8_t *hostData = (uint8_t *)malloc(dumpBytes);
          if (hostData != nullptr) {
            aclError cpRet = aclrtMemcpy(hostData, dumpBytes, (void *)devAddr,
                                         dumpBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (cpRet == ACL_SUCCESS) {
              fprintf(fp, "tensor[%zu] data (first %zu bytes):", t, dumpBytes);
              for (size_t b = 0; b < dumpBytes; b++) {
                fprintf(fp, " %02x", hostData[b]);
              }
              fprintf(fp, "\n");
            }
            free(hostData);
          }
        }
      }
      fclose(fp);
    }
  }

  free(tensorInfos);
  return 0;
}

static void TileLangExceptionDumpCallback(aclrtExceptionInfo *exceptionInfo) {
  if (exceptionInfo == nullptr) {
    return;
  }

  void *devArgsPtr = nullptr;
  uint32_t devArgsLen = 0;
  aclError ret =
      aclrtGetArgsFromExceptionInfo(exceptionInfo, &devArgsPtr, &devArgsLen);
  if (ret != ACL_SUCCESS || devArgsPtr == nullptr || devArgsLen == 0) {
    return;
  }

  uint8_t *hostArgs = (uint8_t *)malloc(devArgsLen);
  if (hostArgs == nullptr) {
    return;
  }

  ret = aclrtMemcpy(hostArgs, devArgsLen, devArgsPtr, devArgsLen,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    free(hostArgs);
    return;
  }

  tilelang_dump_from_host_args(hostArgs, devArgsLen);
  free(hostArgs);
}

static inline void tilelang_register_exception_dump_callback() {
  static bool registered = false;
  if (!registered) {
    aclrtSetExceptionInfoCallback(TileLangExceptionDumpCallback);
    registered = true;
  }
}

#ifdef __cplusplus
}
#endif

#endif
