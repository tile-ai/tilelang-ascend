/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <fstream>
#include <algorithm>

#include "runtime/runtime/rt.h"

// Use map to differentiate same name functions from different binary
static std::unordered_map<std::string, size_t> registered_names;
static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;

static std::tuple<void *, void *>
registerKernel(const char *name, const void *data, size_t data_size, int shared,
               int device, const char *kernel_mode_str) {
  rtError_t rtRet;

  rtDevBinary_t devbin;
  devbin.data = data;
  devbin.length = data_size;
  const std::string kernel_mode{kernel_mode_str};
  if (kernel_mode == "aiv")
    devbin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  else
    devbin.magic = RT_DEV_BINARY_MAGIC_ELF;
  devbin.version = 0;

  rtRet = rtSetDevice(device);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtSetDevice failed, 0x%x\n", rtRet);
    return {NULL, NULL};
  }

  void *devbinHandle = NULL;
  rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtDevBinaryRegister failed, 0x%x\n", rtRet);
    return {NULL, NULL};
  }

  std::string stubName = name;
  stubName += "_" + std::to_string(registered_names[name]);
  registered_names[name]++;
  auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
  void *func_stub_handle = registered.first->second.get();
  rtRet = rtFunctionRegister(devbinHandle, func_stub_handle, stubName.c_str(),
                             (void *)name, 0);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtFunctionRegister failed(stubName = %s), 0x%x\n", stubName.c_str(),
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
  char name[64] = {'\0'};

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

static PyObject *createStream(PyObject *self, PyObject *args) {
  rtStream_t stream;  
  rtError_t rtRet = rtStreamCreate(&stream, 0);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtStreamCreate failed, 0x%x", rtRet);
  	return NULL;
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  uint64_t stream_uint64 = reinterpret_cast<uint64_t>(stream);
  PyObject* result = Py_BuildValue("K", stream_uint64);
  if (result == NULL) {
    rtStreamDestroy(stream);
  }
  return result;
}

/**
 * Read binary data from a file into a vector.
 *
 * @param filename Path to the binary file
 * @return Vector of floats read from the file
 * @throws std::runtime_error if file cannot be opened or read
 */
std::vector<char> readDataFromBinaryFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  file.seekg(0, std::ios::end);
  const size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);  
  // const size_t count = fileSize / sizeof(float);
  // if (fileSize % num_bytes_in_elem != 0) {
  // 	throw std::runtime_error("File size is not a multiple of float size");
  // }  
  // Read the data into a vector
  std::vector<char> data(fileSize);
  file.read(data.data(), fileSize);
  // Check if the read was successful
  if (!file) {
    throw std::runtime_error("Failed to read entire file");
  }
  return data;
}

static PyObject *readDataFromBinaryFileWrapper(PyObject *self, PyObject *args) {
  const char *filename;
  uint64_t arr_ptr;
  if (!PyArg_ParseTuple(args, "sK", &filename, &arr_ptr)) {
  	return NULL;
  }  
  try {
    std::vector<char> data = readDataFromBinaryFile(filename);
    char *arr = reinterpret_cast<char *>(arr_ptr);
    std::copy(data.begin(), data.end(), arr);
    return Py_None;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

void writeDataToBinaryFile(const std::string& filename, const char* data, size_t num_bytes) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  file.write(data, num_bytes);
  if (!file) {
    throw std::runtime_error("Failed to write to file");
  }
}

static PyObject *writeDataToBinaryFileWrapper(PyObject *self, PyObject *args) {
  const char *filename;
  uint64_t arr_ptr;
  size_t num_bytes;    
  if (!PyArg_ParseTuple(args, "sKn", &filename, &arr_ptr, &num_bytes)) {
    return NULL;
  }    
  try {
    const char* data = reinterpret_cast<const char*>(arr_ptr);
    writeDataToBinaryFile(filename, data, num_bytes);
    return Py_None;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

static PyObject* allocateHostMemory(PyObject* self, PyObject* args) {
	uint64_t num_bytes;
	if (!PyArg_ParseTuple(args, "K", &num_bytes)) {
		return NULL;
	}

	void* host_ptr = NULL;
	rtError_t error = rtMallocHost(&host_ptr, num_bytes, RT_MEMORY_HOST);
	if (error != RT_ERROR_NONE) {
		PyErr_Format(PyExc_RuntimeError, "rtMallocHost failed with error code: 0x%x", error);
		return NULL;
	}
  PyObject* result = Py_BuildValue("K", (uint64_t)host_ptr);
  if (result == NULL) {
    rtFreeHost(host_ptr);
  }
	return result;
}

static PyObject* allocateDeviceMemory(PyObject* self, PyObject* args) {
  uint64_t num_bytes;
  if (!PyArg_ParseTuple(args, "K", &num_bytes)) {
    return NULL;
  }
  void* device_ptr = NULL;
  rtError_t error = rtMalloc(&device_ptr, num_bytes, RT_MEMORY_HBM, 0);
  if (error != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtMalloc failed with error code: 0x%x", error);
    return NULL;
  }
  PyObject* result = Py_BuildValue("K", (uint64_t)device_ptr);
  if (result == NULL) {
    rtFree(device_ptr);
  }
  return result;
}

static PyObject* copyMemory(PyObject* self, PyObject* args) {
  uint64_t dst_ptr;
  uint64_t src_ptr;
  size_t count;
  const char* direction_str;
  rtMemcpyKind_t copy_direction;
  if (!PyArg_ParseTuple(args, "KKns", &dst_ptr, &src_ptr, &count, &direction_str)) {
    return NULL;
  }
  if (strcmp(direction_str, "H2D") == 0) {
    copy_direction = RT_MEMCPY_HOST_TO_DEVICE;
  } else if (strcmp(direction_str, "D2H") == 0) {
    copy_direction = RT_MEMCPY_DEVICE_TO_HOST;
  } else {
    PyErr_SetString(PyExc_ValueError, "Invalid copy direction. Must be 'H2D' or 'D2H'.");
    return NULL;
  }
  void *dst = (void*)dst_ptr;
  void *src = (void*)src_ptr;
  rtError_t error = rtMemcpy(dst, count, src, count, copy_direction);
  if (error != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtMemcpy failed with error code: 0x%x", error);
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *getDeviceNum(PyObject *self, PyObject *args) {
  int32_t deviceCount;

  rtError_t rtRet = rtGetDeviceCount(&deviceCount);

  if (rtRet != RT_ERROR_NONE) {
    printf("rtGetDeviceCount failed, 0x%x", rtRet);
    return NULL;
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("I", deviceCount);
}

static PyMethodDef NpuUtilsMethods[] = {
    {"load_kernel_binary", loadKernelBinary, METH_VARARGS, "Load NPU kernel binary into NPU driver"},
    {"get_arch", getArch, METH_VARARGS, "Get soc version of NPU"},
    {"get_aicore_num", getAiCoreNum, METH_VARARGS, "Get the number of AI core"},
    {"get_device_num", getDeviceNum, METH_VARARGS, "Get the number of Ascend device"},
    {"create_stream", createStream, METH_VARARGS, "Create a stream"},
    {"read_data_from_file", readDataFromBinaryFileWrapper, METH_VARARGS, "Read binary file into the array already allocated"},
    {"write_data_to_file", writeDataToBinaryFileWrapper, METH_VARARGS, "Write an array to a binary file"},
    {"allocate_device_memory", allocateDeviceMemory, METH_VARARGS, "Allocate device memory"},
    {"allocate_host_memory", allocateHostMemory, METH_VARARGS, "Allocate host memory"},
    {"copy_memory", copyMemory, METH_VARARGS, "Copy data between host and device"},
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