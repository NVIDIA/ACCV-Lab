# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ctypes

from . import _nvtx_numba_ext as _ext  # type: ignore[attr-defined]

_SYMBOLS_READY = False


def _try_register_numba_symbols() -> bool:
    """
    Register the extension's C symbols with llvmlite so that Numba ``@njit``
    functions can call them.  Returns ``False`` if Numba/llvmlite are not
    installed (they are optional dependencies).
    """
    try:
        import llvmlite.binding as llvm
    except ImportError:
        return False

    lib = ctypes.CDLL(_ext.__file__)
    push = lib.accvlab_nvtx_range_push
    pop = lib.accvlab_nvtx_range_pop
    llvm.add_symbol("accvlab_nvtx_range_push", ctypes.cast(push, ctypes.c_void_p).value)
    llvm.add_symbol("accvlab_nvtx_range_pop", ctypes.cast(pop, ctypes.c_void_p).value)
    return True


_SYMBOLS_READY = _try_register_numba_symbols()


def register_string(name: str) -> int:
    """
    Register a string with NVTX once and return an integer handle.

    Returns 0 if profiler is not attached (the handle is still safe to pass to
    :func:`range_push`, which treats 0 as a no-op).
    """
    return int(_ext.register_string(name))


def range_push(handle: int) -> None:
    """
    Push an NVTX range using a previously-registered handle.

    This function can be called from within Numba ``@njit`` functions.
    """
    _ext.range_push(int(handle))


def range_pop() -> None:
    """
    Pop an NVTX range.

    This function can be called from within Numba ``@njit`` functions.
    """
    _ext.range_pop()


# ---------------------- Numba lowering (CPU @njit) ----------------------

try:
    from llvmlite import ir
    from numba.core import cgutils, types
    from numba.core.errors import TypingError
    from numba.extending import intrinsic, overload
except ImportError:
    pass
else:

    @intrinsic
    def _range_push_intrin(typingctx, handle):
        sig = types.void(handle)

        def codegen(context, builder, signature, args):
            i64 = ir.IntType(64)
            fnty = ir.FunctionType(ir.VoidType(), [i64])
            fn = cgutils.get_or_insert_function(builder.module, fnty, "accvlab_nvtx_range_push")
            arg0 = args[0]
            if arg0.type != i64:
                arg0 = builder.sext(arg0, i64) if arg0.type.width < 64 else builder.trunc(arg0, i64)
            builder.call(fn, [arg0])
            return context.get_dummy_value()

        return sig, codegen

    @intrinsic
    def _range_pop_intrin(typingctx):
        sig = types.void()

        def codegen(context, builder, signature, args):
            fnty = ir.FunctionType(ir.VoidType(), [])
            fn = cgutils.get_or_insert_function(builder.module, fnty, "accvlab_nvtx_range_pop")
            builder.call(fn, [])
            return context.get_dummy_value()

        return sig, codegen

    @overload(range_push, inline="always")
    def _ov_range_push(handle):
        if isinstance(handle, types.Integer):

            if not _SYMBOLS_READY:
                raise TypingError(
                    "NVTX C symbols were not registered with llvmlite. "
                    "This is unexpected — the extension is present but symbol binding failed at import time."
                )

            def impl(handle):
                _range_push_intrin(handle)

            return impl
        return None

    @overload(range_pop, inline="always")
    def _ov_range_pop():
        if not _SYMBOLS_READY:
            raise TypingError(
                "NVTX C symbols were not registered with llvmlite. "
                "This is unexpected — the extension is present but symbol binding failed at import time."
            )

        def impl():
            _range_pop_intrin()

        return impl
