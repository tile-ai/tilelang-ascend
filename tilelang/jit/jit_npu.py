# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union, List
import shutil
import sys
import sysconfig
import pybind11
import torch
import torch_npu
import functools
from ..engine import lower

from tvm import tir
from tvm.tir import PrimFunc


class LaunchThreadExtractor:
    def __init__(self) -> None:
        self.expressions = []

    def visit_thread_extent(self, node):
        if hasattr(node, "body"):
            self.visit_thread_extent(node.body)
        if hasattr(node, "then_case"):
            self.visit_thread_extent(node.then_case)
        if hasattr(node, "else_case"):
            self.visit_thread_extent(node.else_case)
        if hasattr(node, "block"):
            self.visit_thread_extent(node.block)

        if hasattr(node, "attr_key") and node.attr_key == "thread_extent":
            if node.node.thread_tag == self.thread:
                self.expressions.append(node.value)

    def extract(self, node: PrimFunc, thread: str):
        self.thread = thread
        self.visit_thread_extent(node)
        if self.expressions is None:
            return None
        return self.expressions[0]


def replace_by_longest_key(calculate_str, replace_dict):
    sorted_keys = sorted(replace_dict.keys(), key=lambda x: (-len(x), x))
    result = calculate_str
    for key in sorted_keys:
        result = result.replace(key, str(replace_dict[key]))
    return result


def _transform_stmt(stmt, symbolic_var_names):
    """Convert statements, remove symbolic variable definitions"""
    if stmt is None:
        return stmt

    # Handling Block Nodes
    if isinstance(stmt, tir.Block):
        new_body = _transform_stmt(stmt.body, symbolic_var_names)
        return tir.Block(
            iter_vars=stmt.iter_vars,
            reads=stmt.reads,
            writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=new_body,
            init=stmt.init,
            alloc_buffers=stmt.alloc_buffers,
            annotations=stmt.annotations,
            span=stmt.span,
        )

    # Process the SeqStmt node.
    elif isinstance(stmt, tir.SeqStmt):
        new_seqs = []
        for seq_stmt in stmt.seq:
            transformed = _transform_stmt(seq_stmt, symbolic_var_names)
            if transformed is not None:
                new_seqs.append(transformed)
        return tir.SeqStmt(new_seqs, stmt.span) if new_seqs else None

    # Processing LetStmt Node - Removing Symbol Variable Definitions
    elif isinstance(stmt, tir.LetStmt):
        if stmt.var.name in symbolic_var_names:
            # Skip symbol variable definitions, only transform the body.
            return _transform_stmt(stmt.body, symbolic_var_names)
        else:
            # Convert a regular LetStmt
            new_body = _transform_stmt(stmt.body, symbolic_var_names)
            return tir.LetStmt(
                var=stmt.var, value=stmt.value, body=new_body, span=stmt.span
            )

    # Other types of statements remain unchanged.
    else:
        return stmt


# Collect all variables used in the buffer shape
def _process_dynamic_symbolic(func):
    params = func.params
    buffer_map = func.buffer_map
    dynamic_symbolic_map = {}
    for i, param in enumerate(params):
        if param not in buffer_map:
            continue
        buffer = buffer_map[param]
        for j, shape in enumerate(buffer.shape):
            if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                dynamic_symbolic_map[shape] = (i, j)
    return dynamic_symbolic_map


# Creating a Pass for Symbolic Variable Promotion
def _symbolic_var_promoter_pass(func: PrimFunc):
    dynamic_symbolic_map = _process_dynamic_symbolic(func)
    symbolic_vars = list(dynamic_symbolic_map.keys())

    if len(symbolic_vars) == 0:
        return func, {}

    # Create new parameter list: symbolic variables + original parameters
    new_params = list(func.params) + symbolic_vars

    # Transform function body, remove symbolic variable definitions
    new_body = _transform_stmt(func.body, symbolic_vars)

    # Create a new PrimFunc
    new_primfunc = tir.PrimFunc(
        params=new_params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
        span=func.span,
    )
    return new_primfunc, dynamic_symbolic_map


def _get_npucompiler_path() -> str:
    # Set the environment variables for the compiler
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home is None:
        raise Exception("CANN environment not detected (ASCEND_HOME_PATH)")
    bisheng_install_path = os.environ.get("BISHENG_INSTALL_PATH")
    if bisheng_install_path is not None:
        return os.path.join(bisheng_install_path, "bishengir-compile")
    else:
        bishengir = os.path.join(ascend_home, "bisheng_toolkit", "bishengir", "bin")
        return os.path.join(bishengir, "bishengir-compile")


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12,  # BOOL
        # Integer types
        "i8": 2,  # INT8
        "i16": 6,  # INT16
        "i32": 3,  # INT32
        "i64": 9,  # INT64
        # Unsigned integer types
        "u32": 8,  # UINT32
        "u64": 10,  # UINT64
        # Floating point types
        "fp16": 1,  # FLOAT16
        "bf16": 27,  # DT_BF16
        "fp32": 0,  # FLOAT
        "fp64": 11,  # DOUBLE
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")

    return MAP_SIGTYPE_TO_INT[sigty]

def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TILELANG_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bisheng or TILELANG_NPU_COMPILER_PATH"
            )
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path

def extract_device_print_code_from_cann():
    #from triton.backends.ascend.utils import _get_bisheng_path
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib")

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), 'r') as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = ' '.join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = '\n'.join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer(r'\[aicore\]', code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if src[i] == '}':  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == '{':
                    brace_nest += 1
                elif src[j] == '}':
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ''
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace('__gm__', ' ')
        new_code = new_code.replace('__CCELIB_RT_ERROR_NONE', 'RT_ERROR_NONE')
        new_code = new_code.replace('__CCELIB_RT_MEMORY_HBM', 'RT_MEMORY_HBM')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_HOST_TO_DEVICE', 'RT_MEMCPY_HOST_TO_DEVICE')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_DEVICE_TO_HOST', 'RT_MEMCPY_DEVICE_TO_HOST')
        return new_code

    # the following headers should be included in this order
    return '\n'.join([
        read_header('common/common_impl.h'),
        read_header('internal/debug_tunnel/payload.h'),
        read_header('internal/debug_tunnel/payload_impl.h'),
        read_header('internal/debug_tunnel/tunnel.h'),
        read_header('internal/debug_tunnel/tunnel_impl.h')
    ])


def generate_npu_wrapper_src(constants, signature, workspace_size, mix_mode, lock_num, lock_ini_val):
    def _ty_to_cpp(ty):
        if ty[0] == "*":
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _extracted_ty(ty):
        if ty[0] == "*":
            return "PyObject*"
        return {
            "i1": "int32_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gridX, gridY, gridZ;
        rtStream_t stream;
        const void *function;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + "".join(
        [_format_of(_extracted_ty(ty)) for ty in signature.values()]
    )

    grid_info = {"X": "i32", "Y": "i32", "Z": "i32"}

    enable_taskqueue = os.getenv("TILELANG_ENABLE_TASKQUEUE", "true").lower() in (
        "true",
        "1",
    )
    enable_auto_map_parallel_blocks = False
    npu_utils = NPUUtils()
    num_physical_blocks = (
        npu_utils.get_aivector_core_num()
        if mix_mode == "aiv"
        else npu_utils.get_aicore_num()
    )
    task_type = (
        "MSPROF_GE_TASK_TYPE_AIV"
        if mix_mode == "aiv"
        else "MSPROF_GE_TASK_TYPE_AI_CORE"
    )
    LINE_CHANGE_CHAR = chr(10)  # it is \n

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {LINE_CHANGE_CHAR.join(
          f'dataTypes[{i}] = {convert_sigtype_to_int(ty[1:])};'
          for i, ty in signature.items()
          if ty.startswith("*") and i < 5
        )}
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string>
#include <sys/syscall.h>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
{'#include <torch_npu/csrc/framework/OpCommand.h>' if enable_taskqueue else ''}
#include "experiment/runtime/runtime/rt.h"
{extract_device_print_code_from_cann()}

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds, {arg_decls}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattened 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  {'auto launch_call = [=]()' if enable_taskqueue else ''} {{
    uint32_t blockNum = gridX * gridY * gridZ;
    {'blockNum = std::min(blockNum, (uint32_t)' + str(num_physical_blocks) + ');' if enable_auto_map_parallel_blocks else ''}
    cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);
    rtError_t ret;
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    // stub argument for workspace
    void *syncBlockLock = NULL;
    void *workspace_addr = NULL;
    uint16_t ModuleId = 0;
    {f'''
    uint64_t syncBlockLockSize = {lock_num} * sizeof(int64_t);
    ret = rtMalloc(reinterpret_cast<void **>(&syncBlockLock),
                   syncBlockLockSize, RT_MEMORY_HBM, 0);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    std::vector<int64_t> lockInitData({lock_num}, {lock_ini_val});
    ret = rtMemcpy(syncBlockLock, syncBlockLockSize, reinterpret_cast<void *>(lockInitData.data()),
                   syncBlockLockSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if lock_num > 0 else ''}
    {f'''
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    ret = rtMalloc(reinterpret_cast<void **>(&workspace_addr),
                   totalWorkSpaceSize, RT_MEMORY_HBM, ModuleId);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if workspace_size > 0 else ''}
    struct __attribute__((packed)) {{
      void* ffts_addr __attribute__((aligned(8)));
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants)}
      {' '.join(f'{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
      void* DTData __attribute__((aligned(8)));
    }} args = {{
      static_cast<void*>(ffts_addr),
      static_cast<void*>(syncBlockLock),
      static_cast<void*>(workspace_addr),
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants)},
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
      , static_cast<void*>(DTData)
    }};
    {cpp_msprof_call_before_launch}
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
    void *&stream_ref = const_cast<void*&>(stream);
    cce::internal::DebugTunnel::Close(DTData, stream_ref);
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else ''}
   }};
   {'at_npu::native::OpCommand::RunOpApi(name.c_str(), launch_call);' if enable_taskqueue else ''}
  return;
}}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {{
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {{
    return shape;
  }}

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {{
    return shape;
  }}
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {{
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {{
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {{
        shape.push_back(PyLong_AsLong(dim));
      }}
    }}
  }}
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}
  if (__MsprofFlagL1)
  {{
    {
      LINE_CHANGE_CHAR.join(
        f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
        for i, ty in signature.items() if ty[0] == "*"
      )
    }
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {{
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {{
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }}
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}" for i, ty in signature.items())});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__tilelang_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___tilelang_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  {cpp_msprof_callback}
  return m;
}}
"""


def generate_npu_utils_src():
    return """
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "experiment/runtime/runtime/rt.h"

// Use map to differentiate same name functions from different binary
static std::unordered_map<std::string, size_t> registered_names;
static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;

static std::tuple<void *, void *>
registerKernel(const char *name, const void *data, size_t data_size, int shared,
               int device, const char *kernel_mode_str) {
  // name: Kernel Name
  // data: Pointer to the kernel binary
  // data_size: The size of binary data
  // shared: Unused
  // device: Target Device ID
  // kernel mod str: Kernel mode string (aiv or others)
  rtError_t rtRet;

  // Create a binary data structure
  rtDevBinary_t devbin;
  devbin.data = data;
  devbin.length = data_size;

  // Set the modulus according to the kernel mode.
  const std::string kernel_mode{kernel_mode_str};
  if (kernel_mode == "aiv")
    devbin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  else
    devbin.magic = RT_DEV_BINARY_MAGIC_ELF;

  // Set version number
  devbin.version = 0;

  rtRet = rtSetDevice(device);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtSetDevice failed, 0x%x\\n", rtRet);
    return {NULL, NULL};
  }

  // Register binary data, obtain handle
  void *devbinHandle = NULL;
  rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtDevBinaryRegister failed, 0x%x\\n", rtRet);
    return {NULL, NULL};
  }

  // Create a unique stub name (avoid naming conflicts)
  std::string stubName = name;
  stubName += "_" + std::to_string(registered_names[name]);
  registered_names[name]++;

  // Perform global mapping and create storage stubs
  auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
  void *func_stub_handle = registered.first->second.get();

  // Register function to associate binary handle with stub
  rtRet = rtFunctionRegister(devbinHandle, func_stub_handle, stubName.c_str(),
                             (void *)name, 0);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtFunctionRegister failed(stubName = %s), 0x%x\\n", stubName.c_str(),
           rtRet);
    return {NULL, NULL};
  }

  return std::make_tuple(devbinHandle, func_stub_handle);
}

static PyObject *loadKernelBinary(PyObject *self, PyObject *args) {
  const char *name;        // kernel name
  const char *data;        // binary pointer
  Py_ssize_t data_size;    // binary size
  int shared;              // shared_memory(meaningless now)
  int device;              // device ID
  const char *kernel_mode; // kernel mode

  if (!PyArg_ParseTuple(args, "ss#iis", &name, &data, &data_size, &shared,
                        &device, &kernel_mode)) {
    return NULL;
  }

  auto [module_handle, func_handle] =
      registerKernel(name, data, data_size, shared, device, kernel_mode);

  uint64_t mod = reinterpret_cast<uint64_t>(module_handle);
  uint64_t func = reinterpret_cast<uint64_t>(func_handle);
  if (PyErr_Occurred()) {
    return NULL;
  }

  return Py_BuildValue("(KKii)", mod, func, 0, 0);
}

static PyObject *getArch(PyObject *self, PyObject *args) {
  char name[64] = {'\\0'};

  rtError_t rtRet = rtGetSocVersion(name, 64);

  if (rtRet != RT_ERROR_NONE) {
    printf("rtGetSocVersion failed, 0x%x", rtRet);
    return NULL;
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("s", name);
}

static PyObject *getAiCoreNum(PyObject *self, PyObject *args) {
  uint32_t aiCoreCnt;

  rtError_t rtRet = rtGetAiCoreCount(&aiCoreCnt);

  if (rtRet != RT_ERROR_NONE) {
    printf("rtGetAiCoreCount failed, 0x%x", rtRet);
    return NULL;
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("I", aiCoreCnt);
}

static PyMethodDef NpuUtilsMethods[] = {
    {"load_kernel_binary", loadKernelBinary, METH_VARARGS,
     "Load NPU kernel binary into NPU driver"},
    {"get_arch", getArch, METH_VARARGS, "Get soc version of NPU"},
    // sentinel
    {"get_aicore_num", getAiCoreNum, METH_VARARGS, "Get the number of AI core"},
    {NULL, NULL, 0, NULL}};

static PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT, "npu_utils",
    "Utilities for fetching NPU device info and preparing kernel binary", -1,
    NpuUtilsMethods};

PyMODINIT_FUNC PyInit_npu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, NpuUtilsMethods);
  return m;
}
"""


def read_binary_file(file_path, mode="rb", chunk_size=None, return_type="bytes"):
    """
    Function to read a binary file

    Parameters:
        file_path (str): Path to the file to be read
        mode (str): File opening mode, defaults to 'rb' (read binary)
        chunk_size (int): If specified, reads the file in chunks of the given size; otherwise reads the entire file
        return_type (str): Return data type, can be 'bytes' or 'bytearray'

    Returns:
        Returns bytes or bytearray object according to return_type parameter
        If chunk_size is specified, returns a generator that yields data chunk by chunk

    Raises:
        FileNotFoundError: When the file does not exist
        IOError: When an error occurs during file reading
    """
    try:
        with open(file_path, mode) as file:
            if chunk_size:
                # Read file in chunks
                def chunk_reader():
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        if return_type == "bytearray":
                            yield bytearray(chunk)
                        else:
                            yield chunk

                return chunk_reader()
            else:
                # Read the entire file in one go
                data = file.read()
                if return_type == "bytearray":
                    return bytearray(data)
                else:
                    return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error occurred while reading the file: {e}")


class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        cache_path = "npu_utils.so"
        import importlib.util

        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device, mix_mode):
        return self.npu_utils_mod.load_kernel_binary(
            name, kernel, shared, device, mix_mode
        )

    @functools.lru_cache()
    def get_device_properties(self, device):
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"num_aicore": num_aic, "num_vectorcore": num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        return self.npu_utils_mod.get_aicore_num()

    @functools.lru_cache()
    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]

class JitKernel_NPU:
    def __init__(self, metadata: dict) -> None:
        # 1 launch path
        self.so_launcher_path = f"{metadata['kernel_name']}.so"
        self.utils_name = f"{metadata['name']}"
        # 2 kernel path
        self.utils_kernel_src = metadata["kernel_src"]
        self.utils_shared = metadata[
            "shared"
        ]  # Retain the interface, temporarily not in effect.
        self.mix_mode = metadata["mix_mode"]
        self.utils_device = torch.npu.current_device()
        self.launch_stream = torch.npu.current_stream(
            torch.npu.current_device()
        ).npu_stream
        self.launch_packedMetadata = {
            "kernel_name": f"{metadata['name']}",
            "tensor_kinds": metadata["tensor_kinds"],
        }
        self.launch_metadata = {}
        self.launch_enter_hook = None
        self.launch_exit_hook = None
        self.gridfunc = metadata["gridfunc"]
        self.symbolic = metadata["symbolic"]
        self._launch()

    def _launch(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "__tilelang_launcher", self.so_launcher_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch_npu = getattr(mod, "launch")

    def _calcu_grid(self, *args: Any):
        dynamic_val = {}
        extra_args = []
        for key, pos in self.symbolic.items():
            # Ensure that pos is a tuple with two elements.
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                tensor_idx, dim_idx = pos[0], pos[1]
                if tensor_idx < len(args) and hasattr(args[tensor_idx], "shape"):
                    if dim_idx < len(args[tensor_idx].shape):
                        value = args[tensor_idx].shape[dim_idx]
                        dynamic_val[str(key)] = value
                        extra_args.append(value)
                    else:
                        raise IndexError(
                            f"Dimension index {dim_idx} out of range for tensor {tensor_idx}"
                        )
                else:
                    raise IndexError(
                        f"Tensor index {tensor_idx} out of range or tensor has no shape attribute"
                    )
            else:
                raise ValueError(f"Invalid position format for key '{key}': {pos}")
        self.extra_args = extra_args
        result = replace_by_longest_key(self.gridfunc, dynamic_val)

        try:
            # If the result is a number, use it directly.
            if isinstance(result, (int, float)):
                grid_value = result
            # If the result is a string, evaluate it safely.
            elif isinstance(result, str):
                # Only mathematical expressions are allowed to avoid security risks.
                grid_value = eval(
                    result,
                    {"__builtins__": {}},
                    {"math": __import__("math"), **dynamic_val},
                )
            else:
                grid_value = result

            # Ensure the result is an integer
            if hasattr(grid_value, "__iter__"):
                self.launch_grid = [int(x) for x in grid_value]
            else:
                self.launch_grid = [int(grid_value), 1, 1]

        except Exception as e:
            raise ValueError(f"Failed to evaluate grid expression '{result}': {e}")

    def __call__(self, *args: Any) -> Any:
        npu_utils = NPUUtils()
        t_module, t_function, t_n_regs, t_n_spills = npu_utils.load_binary(
            self.utils_name,
            self.utils_kernel_src,
            self.utils_shared,
            self.utils_device,
            self.mix_mode,
        )
        self._calcu_grid(*args)
        return self.launch_npu(
            self.launch_grid[0],
            self.launch_grid[1],
            self.launch_grid[2],
            self.launch_stream,
            t_function,
            self.launch_packedMetadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *args,
            *self.extra_args
        )


class compiler_npu:
    def __init__(self) -> None:
        pass

    def compile(self, mod: PrimFunc) -> JitKernel_NPU:
        self.metadata = {}
        self.mod, self.metadata["symbolic"] = _symbolic_var_promoter_pass(mod)
        # get grid message
        self._parse_grid()
        mlir_path = lower(self.mod)
        if mlir_path.endswith(".mlir"):
            self.mlir_content = self._read_mlir_file(mlir_path)
        else:
            self.mlir_content = mlir_path
        self.constants = {}
        # get signature information
        self.signature = self._parse_signature()
        # TODO: Will be implemented as automatic derivation in the future
        self.workspace_size = 32768
        TILELANG_ASCEND_WORKSPACE_SIZE = os.environ.get('TILELANG_ASCEND_WORKSPACE_SIZE')
        if not TILELANG_ASCEND_WORKSPACE_SIZE is None:
          try:
              self.workspace_size = int(TILELANG_ASCEND_WORKSPACE_SIZE)
          except ValueError:
              print(f"Warning: TILELANG_ASCEND_WORKSPACE_SIZE must be integer, \
                    got '{TILELANG_ASCEND_WORKSPACE_SIZE}', using default 32768")
        self.lock_num = -1
        self.lock_ini_val = 0
        self._parse_npuir_metadata()
        self.metadata["kernel_src"] = self._npuir_to_bin_enable_npu_compile()
        self.wrapper_utiles = generate_npu_utils_src()
        self.so_utils_path = self.make_npu_launcher_stub(
            "npu_utils", self.wrapper_utiles
        )
        self.wrapper_src = generate_npu_wrapper_src(
            self.constants,
            self.signature,
            self.workspace_size,
            self.metadata["mix_mode"],
            self.lock_num,
            self.lock_ini_val,
        )
        self.so_launcher_path = self.make_npu_launcher_stub(
            self.metadata["kernel_name"], self.wrapper_src
        )
        return JitKernel_NPU(metadata=self.metadata)

    def _parse_grid(self):
        launcher = LaunchThreadExtractor()
        expr = launcher.extract(self.mod, "blockIdx.x")
        self.metadata["gridfunc"] = str(expr)

    def _read_mlir_file(self, file_path) -> str:
        """
        Read the content of the MLIR file and return it as a string.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: File '{file_path}' does not exist")
            return None
        except Exception as e:
            print(f"Error occurred while reading the file: {e}")
            return None

    def _parse_npuir_metadata(self) -> None:
        """
        Parse NPU IR to extract metadata required for NPU compilation.
        Extracts and updates the following fields in metadata:
          - mix_mode
          - kernel_name
          - tensor_kinds (currently hardcoded)
          - shared (currently hardcoded)
          - name (combined kernel_name and mix_mode)

        Additionally, removes the mix_mode attribute from the IR.
        """
        # --- Regular expressions and examples ---
        # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
        KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

        # Example：hivm.module_core_type<MIX> -> MIX
        MIX_MODE_REGEX = r"#hivm\.module_core_type<([^>]+)>"

        # Example: test_mix_aic -> test
        MIX_SUFFIX_REGEX = r"_(mix_aic|mix_aiv)$"

        # Note: Compiled Kernel requires to estimate size of shared memory to occupy
        # Currently, NPU backend does not limit on shared memory
        self.metadata["shared"] = 1
        # the mix mode is also encoded into metadata['name'] for runtime to distinguish
        kernel_name = re.search(KERNEL_NAME_REGEX, self.mlir_content).group(1)
        self.metadata["kernel_name"] = kernel_name
        # matching the end of the _mix_aic or _mix_aiv
        self.metadata["name"] = re.sub(MIX_SUFFIX_REGEX, "", kernel_name)
        self.metadata["tensor_kinds"] = []
        self.metadata["mix_mode"] = (
            re.search(MIX_MODE_REGEX, self.mlir_content).group(1).lower()
        )

    def _parse_signature(self) -> dict:
        """
        Parse parameter types from MLIR text and return a dictionary.
        """
        # Define the data types of concern
        target_types = {
            "i1",
            "i8",
            "i16",
            "i32",
            "i64",
            "u32",
            "u64",
            "fp16",
            "bf16",
            "fp32",
            "f32",
            "fp64",
            "f16",
        }

        # Extract the function signature part (the content within the parentheses)
        pattern = r"func\.func\s*@[^(]*\(([^)]*)\)"
        match = re.search(pattern, self.mlir_content)

        if not match:
            return {}

        params_str = match.group(1)

        # Segmentation parameters
        params = []
        current_param = ""
        brace_count = 0
        angle_count = 0

        for char in params_str:
            if char == "," and brace_count == 0 and angle_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "<":
                    angle_count += 1
                elif char == ">":
                    angle_count -= 1

        if current_param:
            params.append(current_param.strip())

        result = {}
        index = 0

        # Skip parameters insert by compiler
        for param in params[3: -6]:
            # Check if the type includes the target type
            found_type = None
            for t_type in target_types:
                # Check for types with an x prefix (e.g., xf16)
                x_pattern = r"\bx" + t_type + r"\b"
                if re.search(x_pattern, param):
                    found_type = "*" + t_type
                    break
                # Check the common type (such as i32)
                elif re.search(r"\b" + t_type + r"\b", param):
                    found_type = t_type
                    break

            if found_type:
                # Special handling: f16 should be mapped to fp16,
                # and f32 should be mapped to fp32.
                if found_type == "f16":
                    found_type = "fp16"
                elif found_type == "*f16":
                    found_type = "*fp16"
                elif found_type == "f32":
                    found_type = "fp32"
                elif found_type == "*f32":
                    found_type = "*fp32"

                result[index] = found_type
                index += 1

        return result

    def _npuir_to_bin_enable_npu_compile(self):
        linalg = self.mlir_content
        metadata = self.metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            ttadapter_path = os.path.join(tmpdir, "kernel.npuir")
            Path(ttadapter_path).write_text(linalg)
            bin_file = os.path.join(tmpdir, "kernel")
            bin_path = os.path.join(tmpdir, "kernel.o")

            npu_compiler_path = _get_npucompiler_path()
            # TileLang Ascend JIT Runtime now follows Triton JIT style.
            # bishengir-compile --enable-triton-kernel-compile=true make sure the way.
            _compile_option_list = [
                "--enable-auto-multi-buffer=true",
                "--enable-triton-kernel-compile=true",
                "--enable-hivm-compile=true"
            ]

            TILELANG_ASCEND_MODE = os.environ.get('TILELANG_ASCEND_MODE')
            if TILELANG_ASCEND_MODE is None:
                _compile_option_list.append("--disable-hivm-tensor-compile=true")
            elif TILELANG_ASCEND_MODE.lower().strip() in ['expert', 'exp', 'e']:
                _compile_option_list.append("--disable-hivm-tensor-compile=true")

            cmd_list = (
                [npu_compiler_path, ttadapter_path]
                + _compile_option_list
                + ["-o", bin_file]
            )
            try:
                ret = subprocess.run(
                    cmd_list, capture_output=True, check=True, text=True
                )
                print("AscendNPU IR compile success:", ret.stdout)
            except subprocess.CalledProcessError as e:
                # print ir
                print("AscendNPU IR:\n")
                print(self.mlir_content)
                # print error info
                print("err cmd:", " ".join(cmd_list))
                print(f"err code: {e.returncode}")
                print("err info:", e.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"error: {str(e)}")
                sys.exit(1)
            return Path(bin_path).read_bytes()

    def make_npu_launcher_stub(self, name: str, source: str, debug=False):
        """
        Generate the launcher stub to launch the kernel
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{name}.cxx")
            with open(src_path, "w") as f:
                f.write(source)
            so = self._build_npu_ext(name, src_path, tmpdir, kernel_launcher="torch")
            return so

    def _get_ascend_path(self):
        return os.environ.get("ASCEND_HOME_PATH")

    def _check_cxx11_abi(self):
        import torch
        return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

    def _build_npu_ext(
        self, obj_name: str, src_path, src_dir, *, kernel_launcher=None
    ) -> str:
        so_path = f"{obj_name}.so"
        cxx = os.environ.get("CC")
        if cxx is None:
            clangxx = shutil.which("clang++")
            gxx = shutil.which("g++")
            cxx = clangxx if clangxx is not None else gxx
            if cxx is None:
                raise RuntimeError("Failed to find C++ compiler")
        cc_cmd = [cxx, src_path]
        # disable all warnings
        cc_cmd += [f"-w"]
        # find the python library
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
        cc_cmd += [f"-I{py_include_dir}"]
        # device_print.h
        cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
        # find the ascend library
        asc_path = self._get_ascend_path()

        cc_cmd += [
            f"-I{os.path.join(asc_path, 'include')}",
            f"-I{os.path.join(asc_path, 'include/experiment')}",
            f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
            f"-I{pybind11.get_include()}",
            f"-L{os.path.join(asc_path, 'lib64')}",
            "-lruntime",
            "-lascendcl",
        ]

        if kernel_launcher == "torch":
            torch_path = os.path.dirname(os.path.realpath(torch.__file__))
            torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
            use_cxx11_abi = self._check_cxx11_abi()
            cc_cmd += [
                f"-I{os.path.join(torch_path, 'include')}",
                f"-I{os.path.join(torch_npu_path, 'include')}",
                f"-L{os.path.join(torch_npu_path, 'lib')}",
                "-ltorch_npu",
                f"-D_GLIBCXX_USE_CXX11_ABI={use_cxx11_abi}",
            ]

        cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", so_path]

        ret = subprocess.check_call(cc_cmd)

        if ret == 0:
            return so_path
        else:
            raise RuntimeError("Failed to compile " + src_path)
