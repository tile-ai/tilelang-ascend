import os
import sys

_env_target = os.environ.get("TILELANG_JIT_TARGET", "").strip()
if _env_target and _env_target != "auto":
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import inspect as _inspect
    import types
    from typing import Sequence, Optional

    _NOT_PASSED = object()

    def _patch_tilelang_jit(mod: types.ModuleType) -> None:
        JitImpl = getattr(mod, "_JitImplementation", None)
        if JitImpl is None:
            return
        if getattr(JitImpl, "__patched_by_env_target__", False):
            return
        orig_init = JitImpl.__init__

        try:
            _sig = _inspect.signature(orig_init)
        except (ValueError, TypeError):
            _sig = None

        if _sig is not None:
            params = list(_sig.parameters.values())
            param_names = [p.name for p in params]
            try:
                _target_pos = param_names.index("target")
            except ValueError:
                _target_pos = -1
        else:
            _target_pos = -1

        def _patched_init(self, *args, **kwargs):
            target_val = _NOT_PASSED
            _pos_check = _target_pos - 1
            if _pos_check >= 0:
                if len(args) > _pos_check:
                    target_val = args[_pos_check]
                elif "target" in kwargs:
                    target_val = kwargs["target"]

            if target_val is _NOT_PASSED or target_val == "auto":
                if _pos_check >= 0 and len(args) > _pos_check:
                    args = tuple(_env_target if i == _pos_check else a
                                 for i, a in enumerate(args))
                else:
                    kwargs["target"] = _env_target

            orig_init(self, *args, **kwargs)

        setattr(_patched_init, "__wrapped__", orig_init)
        JitImpl.__init__ = _patched_init
        JitImpl.__patched_by_env_target__ = True

    class _TilelangJitFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Optional[Sequence[str]],
            target: Optional[types.ModuleType] = None,
        ) -> Optional[importlib.machinery.ModuleSpec]:
            if fullname != "tilelang.jit":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is not None and spec.loader is not None:
                orig_exec = spec.loader.exec_module

                def _wrapped_exec_module(mod):  # type: ignore[no-untyped-def]
                    orig_exec(mod)
                    _patch_tilelang_jit(mod)

                spec.loader.exec_module = _wrapped_exec_module  # type: ignore[assignment]
            return spec

    sys.meta_path.insert(0, _TilelangJitFinder())
