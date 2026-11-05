"""Optional native bindings for the separately compiled A2/A3 backend."""

from tvm import ffi


def register_optional_object(type_key, constructor):
    """Bind a native type only when its compiled constructor is present.

    An OFF build has neither constructor nor type key. Do not invent a native
    key or catch registration errors when an ON constructor is available.
    """

    def register(cls):
        if ffi.get_global_func(constructor, allow_missing=True) is None:
            return cls
        return ffi.register_object(type_key)(cls)

    return register


def require_native(name, operation):
    """Resolve a backend FFI or diagnose its missing compilation component."""
    function = ffi.get_global_func(name, allow_missing=True)
    if function is None:
        raise RuntimeError(f"A2/A3 {operation} requires native components built with USE_ASCEND=ON; missing {name}.")
    return function
