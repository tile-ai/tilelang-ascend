# CMakeLists.txt Debug Configuration

## Understanding the Requirement

To debug C++ code in TileLang, the project must be compiled with:
- Debug symbols (`-g`)
- No optimizations (`-O0`)

This allows GDB to properly inspect variables and step through code.

## Step 1: Locate CMakeLists.txt

The file is located at: `tilelang-ascend/CMakeLists.txt`

## Step 2: Find the Target Library

Search for the line:
```cmake
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})
```

## Step 3: Add Debug Compilation Options

Add the following line immediately after the `add_library` line:
```cmake
target_compile_options(tilelang_objs PRIVATE -g -O0)
```

## Complete Example

```cmake
# ... previous content ...

add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})
target_compile_options(tilelang_objs PRIVATE -g -O0)

# ... rest of the file ...
```

## Step 4: Rebuild the Project

After modifying CMakeLists.txt, rebuild the project to apply the changes:

```bash
cd build
cmakeake clean
cmake ..
make -j$(nproc)
```
