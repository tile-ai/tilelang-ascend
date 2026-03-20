#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include <iostream>
#include <cmath>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

using aclrtStream = void *;
using CallFlashAttention =
    void (*)(uint8_t * /*Q_handle*/, uint8_t * /*K_handle*/,
             uint8_t * /*V_handle*/, uint8_t * /*Output_handle*/,
             uint8_t * /*workspace_1_handle*/, uint8_t * /*workspace_2_handle*/,
             uint8_t * /*workspace_3_handle*/, aclrtStream) noexcept;

// .so wrapper
class DynamicLib {
public:
    DynamicLib() = default;

    // no copy
    DynamicLib(const DynamicLib&) = delete;
    DynamicLib& operator=(const DynamicLib&) = delete;

    // enable move
    DynamicLib(DynamicLib&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    DynamicLib& operator=(DynamicLib&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    // auto close
    ~DynamicLib() {
        close();
    }

    bool load(const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            handle_ = dlopen(path.c_str(), RTLD_LAZY);
            if (handle_) {
                std::cout << "Loaded libop.so from: " << path << std::endl;
                return true;
            }
        }
        std::cerr << "Failed to load libop.so: " << dlerror() << std::endl;
        return false;
    }

    template<typename T>
    T get_symbol(const char* name) const {
        if (!handle_) return nullptr;
        return reinterpret_cast<T>(dlsym(handle_, name));
    }

    bool is_loaded() const {
        return handle_ != nullptr;
    }

private:
    void close() noexcept {
        if (handle_) {
            ::dlclose(handle_);
            handle_ = nullptr;
        }
    }

    void* handle_ = nullptr;
};


template<typename FuncType>
class OpLib {
public:
    bool load(const std::vector<std::string>& paths = {"./libop.so", "../libop.so", "./lib/libop.so", "../lib/libop.so"}) {
        if (!lib_.load(paths)) {
            return false;
        }
        func_ = lib_.get_symbol<FuncType>("call");
        if (!func_) {
            std::cerr << "Failed to find symbol 'call': " << dlerror() << std::endl;
            return false;
        }
        return true;
    }

    FuncType get_func() const {
        return func_;
    }

private:
    DynamicLib lib_;
    FuncType func_ = nullptr;
};

at::Tensor flash_attention_fwd(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V) {
    // lazy loading with static
    static auto* op_lib = []() -> auto {
        static OpLib<CallFlashAttention> lib;
        TORCH_CHECK(lib.load(), "flash_attention_fwd: Failed to load libop.so");
        return &lib;
    }();

    static CallFlashAttention call_func = op_lib->get_func();

    constexpr int64_t block_M = 64;
    constexpr int64_t block_N = 64;

    auto dtype = at::kHalf;
    auto accum_dtype = at::kFloat;

    TORCH_CHECK(Q.dim() == 4, "flash_attention_fwd: Q must be a 4-D tensor");
    TORCH_CHECK(K.dim() == 4, "flash_attention_fwd: K must be a 4-D tensor");
    TORCH_CHECK(V.dim() == 4, "flash_attention_fwd: V must be a 4-D tensor");

    // Q, K, V: [batch, heads, seq_len, dim]
    const int64_t batch = Q.size(0);
    const int64_t heads = Q.size(1);
    const int64_t seq_len = Q.size(2);
    const int64_t dim = Q.size(3);

    const int64_t block_num = (seq_len / block_M) * heads * batch;

    at::Tensor Output = at::empty_like(Q);
    at::Tensor workspace_1 = at::empty({block_num, block_M, block_N}, Q.options().dtype(accum_dtype));
    at::Tensor workspace_2 = at::empty({block_num, block_M, block_N}, Q.options().dtype(dtype));
    at::Tensor workspace_3 = at::empty({block_num, block_M, dim}, Q.options().dtype(accum_dtype));

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    TORCH_CHECK(stream != nullptr, "flash_attention_fwd: Get current NPU stream failed.");

    call_func(
        reinterpret_cast<uint8_t*>(Q.data_ptr()),
        reinterpret_cast<uint8_t*>(K.data_ptr()),
        reinterpret_cast<uint8_t*>(V.data_ptr()),
        reinterpret_cast<uint8_t*>(Output.data_ptr()),
        reinterpret_cast<uint8_t*>(workspace_1.data_ptr()),
        reinterpret_cast<uint8_t*>(workspace_2.data_ptr()),
        reinterpret_cast<uint8_t*>(workspace_3.data_ptr()),
        stream
    );
    return Output;
}

at::Tensor ref_flash_attn(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V) {
    // Q, K, V: [batch, heads, seq_len, dim]
    auto Q_float = Q.to(at::kFloat);
    auto K_float = K.to(at::kFloat);
    auto V_float = V.to(at::kFloat);

    const double scale = 1.0 / std::sqrt(static_cast<double>(Q.size(-1)));
    auto acc = at::einsum("bhsd,bhkd->bhsk", {Q_float, K_float}) * scale;
    acc = at::softmax(acc, /*dim=*/ -1);
    auto o = at::einsum("bhsk,bhkd->bhsd", {acc, V_float});
    return o.to(at::kHalf);
}

int main() {
    constexpr int64_t B = 4;
    constexpr int64_t S = 4096;
    constexpr int64_t H = 16;
    constexpr int64_t D = 128;

    torch_npu::init_npu("npu:0");
    auto device = torch::Device("npu:0");

    torch::manual_seed(0);

    auto options = torch::TensorOptions().dtype(at::kHalf).device(device);
    torch::Tensor q = torch::randn({B, H, S, D}, options);
    torch::Tensor k = torch::randn({B, H, S, D}, options);
    torch::Tensor v = torch::randn({B, H, S, D}, options);

    torch::npu::synchronize();
    std::cout << "init successful!" << std::endl;

    torch::Tensor output = flash_attention_fwd(q, k, v);
    torch::Tensor ref_output = ref_flash_attn(q, k, v);

    torch::npu::synchronize();

    TORCH_CHECK(
        torch::allclose(ref_output, output, /*rtol=*/ 1e-2, /*atol=*/ 1e-2),
        "Flash Attention output does NOT match reference!"
    );

    torch_npu::finalize_npu();

    std::cout << "Test Passed!" << std::endl;
    return 0;
}