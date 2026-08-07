/**
 * perf_rope_ascendc.cpp — RoPE AscendC baseline driver.
 *
 * Calls aclnnRotaryPositionEmbedding (CANN ops-transformer/posembedding/
 * rotary_position_embedding) on a single shape.
 *
 * Two modes:
 *   1. Perf mode (default): random-filled tensors, N repeats, no I/O.
 *      Used with msprof for device Task Duration collection.
 *   2. Golden mode (--load-prefix + --dump-output): load x/sin/cos from
 *      binary files, run once, dump output to file. Used by the Python
 *      test harness for accuracy comparison.
 *
 * Usage (perf):
 *   ./perf_rope_ascendc --shape B S N D --mode 0 --dtype float16 --repeats 6
 *
 * Usage (golden):
 *   ./perf_rope_ascendc --shape B S N D --mode 0 --dtype float16 \
 *     --load-prefix /tmp/rope --dump-output /tmp/rope_out.bin
 *
 * mode: 0=half, 1=interleave  (see aclnn_rotary_position_embedding.h;
 *       NOTE: the header comment says 2=interleave but the actual enum
 *       in op_host is 0=half, 1=interleave, 2=quarter, 3=interleave-half)
 * dtype: float16, bfloat16, float32
 *
 * File format: raw little-endian bytes (fp16/bf16 = 2 bytes/elem,
 * fp32 = 4 bytes/elem). load-prefix reads {prefix}_x.bin, {prefix}_sin.bin,
 * {prefix}_cos.bin. dump-output writes a single file.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

// ====================== Helpers ======================

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t sz = 1;
  for (auto d : shape)
    sz *= d;
  return sz;
}

aclDataType DtypeToAcl(const std::string &dtype) {
  if (dtype == "float16")
    return ACL_FLOAT16;
  if (dtype == "bfloat16")
    return ACL_BF16;
  if (dtype == "float32")
    return ACL_FLOAT;
  LOG_PRINT("Error: unsupported dtype '%s'\n", dtype.c_str());
  exit(1);
}

size_t DtypeSize(const std::string &dtype) {
  if (dtype == "float16" || dtype == "bfloat16")
    return 2;
  if (dtype == "float32")
    return 4;
  return 0;
}

// Read a binary file into a host buffer. Returns false on failure.
bool ReadFile(const std::string &path, void *buf, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    LOG_PRINT("Error: cannot open '%s' for reading\n", path.c_str());
    return false;
  }
  f.read(static_cast<char *>(buf), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(f.gcount()) != bytes) {
    LOG_PRINT("Error: short read on '%s' (got %lld, expected %zu)\n",
              path.c_str(), static_cast<long long>(f.gcount()), bytes);
    return false;
  }
  return true;
}

// Write a host buffer to a binary file. Returns false on failure.
bool WriteFile(const std::string &path, const void *buf, size_t bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    LOG_PRINT("Error: cannot open '%s' for writing\n", path.c_str());
    return false;
  }
  f.write(static_cast<const char *>(buf), static_cast<std::streamsize>(bytes));
  return f.good();
}

// Fill host buffer with random values scaled to [-1, 1]
void FillRandom(void *host_data, int64_t elem_count, const std::string &dtype) {
  std::mt19937 gen(0); // fixed seed = 0, same as TileLang side
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  if (dtype == "float32") {
    auto *p = static_cast<float *>(host_data);
    for (int64_t i = 0; i < elem_count; i++)
      p[i] = dist(gen);
  } else if (dtype == "float16") {
    // FP16: use __fp16 via uint16_t storage
    auto *p = static_cast<uint16_t *>(host_data);
    for (int64_t i = 0; i < elem_count; i++) {
      float val = dist(gen);
      // Convert float32 → float16 via bit manipulation (IEEE 754)
      uint32_t bits;
      memcpy(&bits, &val, sizeof(uint32_t));
      uint32_t sign = (bits >> 16) & 0x8000;
      int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
      uint32_t mant = bits & 0x7FFFFF;
      uint16_t fp16;
      if (exp <= 0) {
        fp16 = sign; // underflow → 0
      } else if (exp >= 31) {
        fp16 = sign | 0x7C00; // overflow → inf
      } else {
        fp16 = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
      }
      p[i] = fp16;
    }
  } else if (dtype == "bfloat16") {
    // BF16: truncate float32 high 16 bits
    auto *p = static_cast<uint16_t *>(host_data);
    for (int64_t i = 0; i < elem_count; i++) {
      float val = dist(gen);
      uint32_t bits;
      memcpy(&bits, &val, sizeof(uint32_t));
      p[i] = static_cast<uint16_t>(bits >> 16);
    }
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed: %d\n", ret);
            return ret);
  return 0;
}

// Create aclTensor on device.
// fill_mode: "random" (FillRandom), "zero" (zero-fill), or file path (load).
int CreateAclTensor(const std::vector<int64_t> &shape, aclDataType dataType,
                    size_t type_size, void **deviceAddr, aclTensor **tensor,
                    const std::string &fill_mode = "random") {
  int64_t elem_count = GetShapeSize(shape);
  size_t bytes = static_cast<size_t>(elem_count) * type_size;

  auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed: %d\n", ret);
            return ret);

  std::vector<uint8_t> host_buf(bytes);
  if (fill_mode == "zero") {
    std::memset(host_buf.data(), 0, bytes);
  } else if (fill_mode == "random") {
    std::string dtype_str = (dataType == ACL_FLOAT)     ? "float32"
                            : (dataType == ACL_FLOAT16) ? "float16"
                                                        : "bfloat16";
    FillRandom(host_buf.data(), elem_count, dtype_str);
  } else {
    // fill_mode is a file path
    if (!ReadFile(fill_mode, host_buf.data(), bytes)) {
      return 1;
    }
  }
  ret = aclrtMemcpy(*deviceAddr, bytes, host_buf.data(), bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
            return ret);

  // Strides for contiguous ND tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor =
      aclCreateTensor(shape.data(), static_cast<int32_t>(shape.size()),
                      dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                      static_cast<int32_t>(shape.size()), *deviceAddr);
  return 0;
}

// ====================== Arg parsing ======================

struct Args {
  std::vector<int64_t> shape; // 4D: B S N D
  int64_t mode = 0;           // 0=half, 1=interleave
  std::string dtype = "float16";
  int repeats = 6;
  int32_t device_id = 0;
  std::string load_prefix; // if set: load {prefix}_x/sin/cos.bin
  std::string dump_output; // if set: dump output to this path
};

void PrintUsage() {
  fprintf(stderr,
          "Usage: perf_rope_ascendc --shape B S N D [--mode 0] [--dtype "
          "float16] [--repeats 6] [--device 0]\n"
          "       perf_rope_ascendc --shape B S N D --load-prefix P "
          "--dump-output F\n"
          "  --shape        4 ints (B S N D), e.g. 4 1 64 128\n"
          "  --mode         0=half, 1=interleave (default: 0)\n"
          "  --dtype        float16|bfloat16|float32 (default: float16)\n"
          "  --repeats      kernel launches (default: 6, first is warm-up)\n"
          "  --device       NPU device id (default: 0)\n"
          "  --load-prefix  load {prefix}_x.bin, {prefix}_sin.bin, "
          "{prefix}_cos.bin\n"
          "  --dump-output  run once and dump output to this file\n");
}

Args ParseArgs(int argc, char *argv[]) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--shape" && i + 4 < argc) {
      for (int j = 0; j < 4; j++) {
        a.shape.push_back(std::atoll(argv[++i]));
      }
    } else if (arg == "--mode" && i + 1 < argc) {
      a.mode = std::atoll(argv[++i]);
    } else if (arg == "--dtype" && i + 1 < argc) {
      a.dtype = argv[++i];
    } else if (arg == "--repeats" && i + 1 < argc) {
      a.repeats = std::atoi(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      a.device_id = std::atoi(argv[++i]);
    } else if (arg == "--load-prefix" && i + 1 < argc) {
      a.load_prefix = argv[++i];
    } else if (arg == "--dump-output" && i + 1 < argc) {
      a.dump_output = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage();
      exit(0);
    } else {
      LOG_PRINT("Unknown arg: %s\n", arg.c_str());
      PrintUsage();
      exit(1);
    }
  }
  if (a.shape.size() != 4) {
    LOG_PRINT("Error: --shape requires exactly 4 ints (B S N D)\n");
    PrintUsage();
    exit(1);
  }
  // Golden mode: force repeats=1 (only need one run for output)
  if (!a.dump_output.empty()) {
    a.repeats = 1;
  }
  return a;
}

// ====================== Main ======================

int main(int argc, char *argv[]) {
  Args args = ParseArgs(argc, argv);

  aclrtStream stream;
  auto ret = Init(args.device_id, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init failed: %d\n", ret); return ret);

  aclDataType dtype = DtypeToAcl(args.dtype);
  size_t type_size = DtypeSize(args.dtype);

  // x: [B, S, N, D]  (4D, ND layout)
  // sin/cos: broadcast over N → [B, S, 1, D]
  // out: same as x
  std::vector<int64_t> x_shape = args.shape; // B S N D
  std::vector<int64_t> sc_shape = {args.shape[0], args.shape[1], 1,
                                   args.shape[3]}; // B S 1 D

  // --- Determine fill modes ---
  bool golden = !args.load_prefix.empty();
  std::string x_fill = golden ? (args.load_prefix + "_x.bin") : "random";
  std::string sin_fill = golden ? (args.load_prefix + "_sin.bin") : "random";
  std::string cos_fill = golden ? (args.load_prefix + "_cos.bin") : "random";

  // --- Create tensors ---
  void *x_dev = nullptr;
  void *cos_dev = nullptr;
  void *sin_dev = nullptr;
  void *out_dev = nullptr;
  aclTensor *x = nullptr;
  aclTensor *cos = nullptr;
  aclTensor *sin = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(x_shape, dtype, type_size, &x_dev, &x, x_fill);
  CHECK_RET(ret == 0, return ret);
  ret = CreateAclTensor(sc_shape, dtype, type_size, &cos_dev, &cos, cos_fill);
  CHECK_RET(ret == 0, return ret);
  ret = CreateAclTensor(sc_shape, dtype, type_size, &sin_dev, &sin, sin_fill);
  CHECK_RET(ret == 0, return ret);
  ret = CreateAclTensor(x_shape, dtype, type_size, &out_dev, &out, "zero");
  CHECK_RET(ret == 0, return ret);

  // --- Run kernel N times (no timing, msprof wraps) ---
  for (int i = 0; i < args.repeats; i++) {
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnRotaryPositionEmbeddingGetWorkspaceSize(
        x, cos, sin, args.mode, out, &workspace_size, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("GetWorkspaceSize failed: %d\n", ret);
              return ret);

    void *workspace = nullptr;
    if (workspace_size > 0) {
      ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("workspace malloc failed: %d\n", ret);
                return ret);
    }

    ret = aclnnRotaryPositionEmbedding(workspace, workspace_size, executor,
                                       stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRotaryPositionEmbedding failed: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SyncStream failed: %d\n", ret);
              return ret);

    if (workspace != nullptr)
      aclrtFree(workspace);
    // executor freed internally by aclnn, do not manually free
  }

  // --- Dump output (golden mode) ---
  if (!args.dump_output.empty()) {
    int64_t out_elems = GetShapeSize(x_shape);
    size_t out_bytes = static_cast<size_t>(out_elems) * type_size;
    std::vector<uint8_t> host_buf(out_bytes);
    ret = aclrtMemcpy(host_buf.data(), out_bytes, out_dev, out_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy (D2H) failed: %d\n", ret);
              return ret);
    if (!WriteFile(args.dump_output, host_buf.data(), out_bytes)) {
      return 1;
    }
  }

  // --- Cleanup ---
  aclDestroyTensor(x);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(out);
  aclrtFree(x_dev);
  aclrtFree(cos_dev);
  aclrtFree(sin_dev);
  aclrtFree(out_dev);
  aclrtDestroyStream(stream);
  aclrtResetDevice(args.device_id);
  aclFinalize();

  LOG_PRINT("Test Passed!\n");
  return 0;
}
