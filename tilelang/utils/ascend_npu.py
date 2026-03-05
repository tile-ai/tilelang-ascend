# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# Copied from bitblas
import functools

class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        # TODO: change to use cache, non-fixed directory (Finish before 330)
        fname = "npu_utils.so"
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod
    
    def load_binary(self, name, kernel, shared, device, mix_mode):
        return self.npu_utils_mod.load_kernel_binary(
            name, kernel, shared, device, mix_mode
        )
    
    @functools.lru_cache()
    def get_arch(self):
        # Return Ascend soc version
        return self.npu_utils_mod.get_arch()
    
    @functools.lru_cache()
    def get_aicore_num(self):
        # Return Ascend soc aicore number
        return self.npu_utils_mod.get_aicore_num()
    
    @functools.lru_cache()
    def get_aivector_core_num(self):
        # Return Ascend soc vector core number
        return self.get_aicore_num() * 2
    
    @functools.lru_cache()
    def get_aicube_core_num(self):
        # Return Ascend soc cube core number
        return self.get_aicore_num()
    
    @functools.lru_cache()
    def get_device_num(self):
        # Return Ascend device number
        return self.npu_utils_mod.get_device_num()