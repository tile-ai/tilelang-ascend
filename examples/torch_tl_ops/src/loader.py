"""
TileLang Ascend Operators - Standalone Kernel Loader

Does not depend on tilelang source code, only depends on:
- torch
- torch_npu
"""

import os
import math
import pickle
import torch
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


def replace_by_longest_key(calculate_str: str, replace_dict: dict) -> str:
    """Replace variable names in string"""
    sorted_keys = sorted(replace_dict.keys(), key=lambda x: (-len(x), x))
    result = calculate_str
    for key in sorted_keys:
        result = result.replace(key, str(replace_dict[key]))
    return result


_shared_npu_utils = None


def get_shared_npu_utils():
    """Get shared npu_utils module (singleton)"""
    global _shared_npu_utils
    if _shared_npu_utils is None:
        utils_path = Path(__file__).parent / "utils" / "npu_utils.so"
        if not utils_path.exists():
            raise FileNotFoundError(f"npu_utils.so not found at {utils_path}")
        spec = importlib.util.spec_from_file_location("npu_utils", str(utils_path))
        _shared_npu_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_shared_npu_utils)
    return _shared_npu_utils


class NPUKernelLoader:
    """
    Standalone NPU Kernel Loader
    
    Loads kernels from precompiled files:
    - metadata.pkl: Kernel metadata
    - main.so: Launcher
    - npu_utils.so: Utility library (shared, stored in utils/)
    - kernel.o: Kernel binary (optional, embedded in metadata)
    """
    
    def __init__(self, kernel_dir: str):
        self.kernel_dir = Path(kernel_dir)
        
        # TODO: Workspace derivation is not currently handled.
        # Once the cache mechanism is stable, switch to cache-based mode.
        
        # Load metadata
        metadata_path = self.kernel_dir / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Extract metadata fields
        self.signature = self.metadata.get("signature", {})
        self.out_idx = self.metadata.get("out_idx", None)
        # Keep None, only convert int to list
        if isinstance(self.out_idx, int):
            self.out_idx = [self.out_idx]
        self.param_info = self.metadata.get("param_info", [])
        self.symbolic = self.metadata.get("symbolic", {})
        self.gridfunc = self.metadata.get("gridfunc", "")
        self.kernel_src = self.metadata.get("kernel_src", b"")
        self.kernel_name = self.metadata.get("name", "kernel")
        self.tensor_kinds = self.metadata.get("tensor_kinds", [])
        self.shared = self.metadata.get("shared", 1)
        self.mix_mode = self.metadata.get("mix_mode", False)
        
        # Device info
        self.device = torch.npu.current_device()
        self.stream = torch.npu.current_stream(self.device).npu_stream
        
        # Load main.so (launcher)
        self._load_launcher()
        
        # Load npu_utils.so (utility library, shared)
        self._load_npu_utils()
        
        # Load kernel binary
        self._load_kernel_binary()
    
    def _load_launcher(self):
        """Load main.so launcher"""
        launcher_path = self.kernel_dir / "main.so"
        spec = importlib.util.spec_from_file_location(
            "__tilelang_launcher", str(launcher_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")
    
    def _load_npu_utils(self):
        """Load npu_utils.so utility library (shared)"""
        self.npu_utils = get_shared_npu_utils()
    
    def _load_kernel_binary(self):
        """Load kernel binary to device"""
        kernel_mode = "aicore" if not self.mix_mode else "aiv"
        
        result = self.npu_utils.load_kernel_binary(
            self.kernel_name,
            self.kernel_src,
            0,
            self.device,
            kernel_mode,
        )
        
        self.t_module, self.t_function, self.t_n_regs, self.t_n_spills = result
    
    def _calc_grid(self, orig_to_input: dict, *args):
        """Calculate grid dimensions and dynamic values"""
        dynamic_val = {}
        extra_args = []
        
        for key, pos in self.symbolic.items():
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                tensor_idx, dim_idx = pos[0], pos[1]
                if tensor_idx in orig_to_input:
                    arg_pos = orig_to_input[tensor_idx]
                    arg = args[arg_pos]
                    if isinstance(arg, torch.Tensor) and dim_idx < len(arg.shape):
                        value = arg.shape[dim_idx]
                        dynamic_val[str(key)] = value
                        extra_args.append(value)
                    else:
                        raise ValueError(f"Cannot resolve symbolic {key}")
                else:
                    raise ValueError(f"Symbolic {key} depends on output")
        
        self.extra_args = extra_args
        
        # Calculate grid
        result = replace_by_longest_key(self.gridfunc, dynamic_val)
        
        try:
            if isinstance(result, (int, float)):
                grid_value = result
            elif isinstance(result, str):
                grid_value = eval(
                    result,
                    {"__builtins__": {}},
                    {"math": math, **dynamic_val},
                )
            else:
                grid_value = result
            
            if hasattr(grid_value, "__iter__"):
                self.grid = [int(x) for x in grid_value]
            else:
                self.grid = [int(grid_value), 1, 1]
        except Exception as e:
            raise ValueError(f"Failed to evaluate grid expression '{result}': {e}")
        
        return dynamic_val
    
    def __call__(self, *args) -> Any:
        """Execute kernel"""
        total_params = len(self.param_info)
        num_inputs = total_params - (len(self.out_idx) if self.out_idx is not None else 0)
        
        if len(args) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {len(args)}")
        
        # Build input position mapping
        orig_to_input = {}
        input_pos = 0
        for i, info in enumerate(self.param_info):
            if not info["is_output"]:
                orig_to_input[i] = input_pos
                input_pos += 1
        
        # Calculate grid and dynamic values
        dynamic_val = self._calc_grid(orig_to_input, *args)
        
        # Build complete parameter list
        full_args = [None] * total_params
        input_ptr = 0
        
        for i, info in enumerate(self.param_info):
            if info["is_output"]:
                # Output parameter: create empty tensor
                dtype = info["dtype"]
                shape = []
                for dim in info["shape"]:
                    if isinstance(dim, str):
                        val = dynamic_val.get(dim)
                        if val is None:
                            raise ValueError(f"Missing value for {dim}")
                        shape.append(val)
                    else:
                        shape.append(int(dim))
                
                device = args[0].device if args else torch.device("npu")
                full_args[i] = torch.empty(shape, dtype=dtype, device=device)
            else:
                # Input parameter
                full_args[i] = args[input_ptr]
                input_ptr += 1
        
        # Add extra parameters
        full_args.extend(self.extra_args)
        
        # Execute kernel
        self.launch(
            self.grid[0],
            self.grid[1],
            self.grid[2],
            self.stream,
            self.t_function,
            {"kernel_name": self.kernel_name, "tensor_kinds": self.tensor_kinds},
            {},
            None,  # enter_hook
            None,  # exit_hook
            *full_args
        )
        
        # Return result
        if self.out_idx is None:
            return None
        elif len(self.out_idx) == 1:
            return full_args[self.out_idx[0]]
        else:
            return [full_args[i] for i in self.out_idx]


class KernelRegistry:
    """Kernel registry"""
    
    _kernels: Dict[str, NPUKernelLoader] = {}
    _kernel_dir: Optional[Path] = None
    
    @classmethod
    def set_kernel_dir(cls, kernel_dir: str):
        """Set kernel directory"""
        cls._kernel_dir = Path(kernel_dir)
    
    @classmethod
    def get_kernel(cls, name: str) -> NPUKernelLoader:
        """Get kernel (with cache)"""
        if name not in cls._kernels:
            if cls._kernel_dir is None:
                # Default to kernels directory in package
                cls._kernel_dir = Path(__file__).parent / "kernels"
            
            kernel_path = cls._kernel_dir / name
            if not kernel_path.exists():
                raise ValueError(f"Kernel '{name}' not found at {kernel_path}")
            
            cls._kernels[name] = NPUKernelLoader(str(kernel_path))
        
        return cls._kernels[name]
