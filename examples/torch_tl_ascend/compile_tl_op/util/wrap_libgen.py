from typing import Optional
from functools import wraps
from tilelang.jit.adapter.libgen import LibraryGenerator

def wrap_load_lib(load_lib):
    @wraps(load_lib)
    def wrapper(self: LibraryGenerator, lib_path: Optional[str] = None):
        lib = load_lib(self, lib_path)
        if lib_path:
            self.libpath = lib_path  # remember libpath
        return lib
    return wrapper

LibraryGenerator.load_lib = wrap_load_lib(LibraryGenerator.load_lib)
