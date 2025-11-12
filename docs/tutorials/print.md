# Printf&Dump_Tensor
## Overview
TileLang-ascend introduces new debugging API interfaces: `T.printf` and `T.dump_tensor`. Developers are welcome to use them! Currently, full dump on Ascend is supported. Basic types, pointers, ub_buffer, l1_buffer, l0c_buffer, and global_buffer are all printable.

*Note: `T.printf` and `T.dump_tensor` are device-side debugging tools; for the host side, use Python's built-in print directly.*

## Printf
### Interface Prototype
    ```python
    def printf(format_str: str, *args)
    ```
- `format_str` is the format string used for printing strings, variables, addresses, and other information. It controls the conversion type through format specifiers % and supports strings, decimals, hexadecimal numbers, floating-point numbers, and pointers.
- `*args` is a variable-length argument list with variable types: Depending on the different format strings, the function may require a series of additional parameters. Each parameter contains a value to be inserted, replacing each % tag specified in the format parameter. The number of parameters should match the number of % tags.
    - Format Specifier Functions:
        - %d/%i: Output decimal integer
        - %f: Output floating-point number
        - %x: Output hexadecimal integer (can be used to output address information)
        - %s: Output string
        - %p: Output pointer address (**It is recommended to use %x for address output directly**)
### Usage of printf
```python
# Supports variable arguments
T.printf("fmt %s %d\n", "string", 0x123)
```

## Dump Tensor
Used to dump the content of a specified Tensor, while also supporting printing custom additional information (only supports uint32_t data type), such as printing the current line number, etc.
### Interface Prototype
```python
def dump_tensor(tensor: Buffer, desc: int, dump_size: int, shape_info: tuple=())
```
- The `tensor` is the tensor that needs to be dumped. It supports `ub_buffer`, `l1_buffer`, `l0c_buffer`, and `global_buffer`. There's no need to distinguish between these types; simply input the name of the `tensor`.
- `desc` is user-defined additional information (line numbers or other meaningful numbers).
- `dump_size` is the number of elements to be dumped.
- `shape_info` is the shape information of the input tensor and can be used to format the printed output.
    - When the shape size is larger than the number of elements specified by `dump_size`, elements are printed according to the `shape_info`, with any missing dump data displayed as "-".
    - When the shape size is less than or equal to the number of elements specified by `dump_size`, elements are printed according to the `shape_info`, and any excess dump data beyond the shape dimensions is not displayed.

### Usage of dump_tensor
```python
## ub_buffer、l1_buffer、l0c_buffer、global_buffer
T.printf("A_L1:\n")
T.dump_tensor(A_L1, 111, 64) # l1_buffer

T.printf("B_L1:\n")
T.dump_tensor(B_L1, 222, 64) # l1_buffer

T.printf("C_L0C:\n")
T.dump_tensor(C_L0C, 333, 64) # l0c_buffer

T.printf("a_ub:\n")
T.dump_tensor(a_ub, 444, 64) # ub_buffer

T.printf("A_GLOBAL:\n")
T.dump_tensor(a_global, 555, 64) # global_buffer

## Using shape_info for clearer dumping

T.printf("A_L1:\n")
T.dump_tensor(A_L1, 111, 64, (8, 8)) # l1_buffer

T.printf("B_L1:\n")
T.dump_tensor(B_L1, 222, 64, (8, 9)) # l1_buffer

T.printf("C_L0C:\n")
T.dump_tensor(C_L0C, 333, 64, (8, 7)) # l0c_buffer

T.printf("a_ub:\n")
T.dump_tensor(a_ub, 444, 64, (8, 8)) # ub_buffer

T.printf("A_GLOBAL:\n")
T.dump_tensor(a_global, 555, 64, (8, 8)) # global_buffer

```
### Other Information  
The DumpTensor print results automatically display highly detailed information at the beginning, including:  
- CANN software package version details  
- Timestamp of the CANN software package release  
- Kernel type information  
- Operator details  
- Memory information  
- Data type  
- Location information  

Example print output:
```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1046912, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=111, addr=0, data_type=float16, position=UB, dump_size=32
```