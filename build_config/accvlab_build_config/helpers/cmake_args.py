import os
import re
from pathlib import Path
from typing import List, Optional

from .build_utils import (
    detect_cuda_info,
    format_torch_cuda_arch_list_from_selection,
    resolve_cuda_architecture_selection,
)

# Marker file at the ACCV-Lab monorepo root (see `.nav` in the repository).
_NAV_MARKER = ".nav"
# Must match `.nav` contents after strip (UTF-8); see repository root `.nav`.
_NAV_EXPECTED_CONTENT = "project root"

CUDA_ARCH_STRATEGY_CMAKE = "cmake"
CUDA_ARCH_STRATEGY_TORCH = "torch"
_VALID_CUDA_ARCH_STRATEGIES = frozenset({CUDA_ARCH_STRATEGY_CMAKE, CUDA_ARCH_STRATEGY_TORCH})


def _nav_file_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return text == _NAV_EXPECTED_CONTENT


def _find_repo_root_via_nav(start: Path) -> Optional[Path]:
    """
    Walk upward from ``start`` (file or directory) until a ``.nav`` file exists
    whose contents equal ``project root`` (UTF-8); that directory is the repo root.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(64):
        nav = cur / _NAV_MARKER
        if _nav_file_is_valid(nav):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _parse_bool_env(value: str) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_cpp_standard(value: str) -> str:
    """
    Convert inputs like 'c++17' or '17' to a plain numeric standard '17'
    """
    if value is None:
        return ""
    v = value.strip().lower()
    if v.startswith("c++"):
        v = v.replace("c++", "")
    return v


def _format_cmake_cuda_architectures(archs: List[str], ptx_archs: List[str]) -> List[str]:
    if not ptx_archs:
        return archs

    cmake_archs: List[str] = []
    for arch in archs:
        cmake_archs.append(f"{arch}-real")
    for arch in ptx_archs:
        cmake_archs.append(f"{arch}-virtual")
    return cmake_archs


def _append_cmake_cuda_architectures_args(args: List[str], cuda_info: dict) -> None:
    selection = resolve_cuda_architecture_selection(cuda_info)
    if selection is None:
        return

    cmake_archs = _format_cmake_cuda_architectures(
        selection.architectures,
        selection.ptx_architectures,
    )
    if not cmake_archs:
        return

    args.append(f'-DCMAKE_CUDA_ARCHITECTURES={";".join(cmake_archs)}')


def _append_torch_cuda_arch_list_args(args: List[str], cuda_info: dict) -> None:
    selection = resolve_cuda_architecture_selection(cuda_info)
    if selection is None:
        return

    torch_cuda_arch_list = format_torch_cuda_arch_list_from_selection(selection)
    if not torch_cuda_arch_list:
        return

    args.append(f"-DACCVLAB_TORCH_CUDA_ARCH_LIST={torch_cuda_arch_list}")


def get_project_root() -> Path:
    """Return the ACCV-Lab monorepo root identified by a valid ``.nav`` marker."""
    anchors = (
        ("accvlab_build_config.helpers.cmake_args", Path(__file__)),
        ("current working directory", Path.cwd()),
    )
    for _label, anchor in anchors:
        found = _find_repo_root_via_nav(anchor)
        if found is not None:
            return found
    raise RuntimeError(
        "ACCV-Lab repository root not found: could not locate a valid `.nav` file "
        f"({_NAV_MARKER!r} with UTF-8 text exactly {_NAV_EXPECTED_CONTENT!r} after strip). "
        "Searched upward from this Python module and from the process current working directory. "
        "Build from a full checkout of the repository, run commands with a cwd inside that tree "
        "(e.g. a package directory under `packages/`), or install `accvlab-build-config` editable "
        "from `build_config/` so `cmake_args` resolves from source."
    )


def _build_cmake_args_from_env(cuda_arch_strategy: str) -> List[str]:
    """
    Build a list of -D CMake arguments from environment variables to harmonize
    build configuration across setuptools, external CMake, and scikit-build flows.

    CUDA architecture flags are prepared according to ``cuda_arch_strategy``:

    - ``cmake``: ``-DCMAKE_CUDA_ARCHITECTURES=...``
    - ``torch``: ``-DACCVLAB_TORCH_CUDA_ARCH_LIST=...`` for skbuild packages that
      call ``find_package(Torch)`` (copied to ``TORCH_CUDA_ARCH_LIST`` in CMake)

    If ``CUSTOM_CUDA_ARCHS`` is unset, detected CUDA architectures become real
    targets only when ``nvcc`` reports exact support. Unsupported detections use
    supported PTX targets at or below the detected architecture. When set,
    ``CUSTOM_CUDA_ARCHS`` is passed through unchanged (see the Installation
    Guide).
    """
    args: List[str] = []
    # Always export compile_commands.json for tooling/validation
    args.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

    # DEBUG_BUILD -> CMAKE_BUILD_TYPE (default to Release if not set)
    if _parse_bool_env(os.environ.get("DEBUG_BUILD", "")):
        args.append("-DCMAKE_BUILD_TYPE=Debug")
    else:
        args.append("-DCMAKE_BUILD_TYPE=Release")

    # CPP_STANDARD -> CMAKE_CXX_STANDARD
    cpp_std = os.environ.get("CPP_STANDARD")
    if cpp_std:
        norm = _normalize_cpp_standard(cpp_std)
        if norm:
            args.append(f"-DCMAKE_CXX_STANDARD={norm}")
            args.append(f"-DCMAKE_CUDA_STANDARD={norm}")

    cuda_info = detect_cuda_info()
    if cuda_arch_strategy == CUDA_ARCH_STRATEGY_CMAKE:
        _append_cmake_cuda_architectures_args(args, cuda_info)
    elif cuda_arch_strategy == CUDA_ARCH_STRATEGY_TORCH:
        _append_torch_cuda_arch_list_args(args, cuda_info)

    # VERBOSE_BUILD -> CMAKE_VERBOSE_MAKEFILE
    if _parse_bool_env(os.environ.get("VERBOSE_BUILD", "")):
        args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

    # Aggregate compiler flags
    optimize_level = os.environ.get("OPTIMIZE_LEVEL")
    use_fast_math = _parse_bool_env(os.environ.get("USE_FAST_MATH", ""))
    enable_profiling = _parse_bool_env(os.environ.get("ENABLE_PROFILING", ""))
    verbose_build = _parse_bool_env(os.environ.get("VERBOSE_BUILD", ""))

    cxx_flags: List[str] = []
    cuda_flags: List[str] = []

    if optimize_level:
        cxx_flags.append(f"-O{optimize_level}")
        cuda_flags.append(f"-O{optimize_level}")

    if use_fast_math:
        cxx_flags.append("-ffast-math")
        cuda_flags.append("--use_fast_math")

    if enable_profiling:
        cxx_flags.append("-pg")
        # For nvcc, pass through to host compiler
        cuda_flags.append("-Xcompiler=-pg")

    if verbose_build:
        # Mirror setuptools helper behavior for nvcc ptxas verbosity
        cuda_flags.append("-Xptxas=-v")

    if cxx_flags:
        args.append(f'-DCMAKE_CXX_FLAGS={" ".join(cxx_flags)}')
    if cuda_flags:
        args.append(f'-DCMAKE_CUDA_FLAGS={" ".join(cuda_flags)}')

    return args


def _build_cmake_args_package_scm_version(repo_root: Path) -> List[str]:
    """
    Pass numeric version from setuptools-scm to CMake as a repo-aligned package
    version define (and harmless for CMake projects that ignore the variable).
    """
    from setuptools_scm import get_version  # type: ignore

    v = get_version(
        root=str(repo_root),
        version_scheme="no-guess-dev",
        fallback_version="0.0.0",
    )
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", v)
    numeric = "0.0.0"
    if m:
        numeric = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return [f"-DACCVLAB_PACKAGE_CMAKE_VERSION={numeric}"]


def build_cmake_args(cuda_arch_strategy: str) -> List[str]:
    """Build the full CMake ``-D`` argument list for ACCV-Lab package builds.

    Combines environment-based build flags with a repo-aligned SCM version define.
    Auto-detected CUDA architectures use exact ``nvcc`` real targets when
    supported. Unsupported detections fall back to supported PTX targets at or
    below the detected architecture when ``CUSTOM_CUDA_ARCHS`` is unset.

    Args:
        cuda_arch_strategy: Select how CUDA GPU architectures are passed to CMake.
            Pass ``CUDA_ARCH_STRATEGY_CMAKE`` for native CMake CUDA targets, or
            ``CUDA_ARCH_STRATEGY_TORCH`` for skbuild packages that call
            ``find_package(Torch)``. Each package's ``setup.py`` must choose the
            strategy appropriate to its build.

    Returns:
        List[str]: CMake ``-D`` arguments derived from environment variables and
        the repository SCM version.
    """
    if cuda_arch_strategy not in _VALID_CUDA_ARCH_STRATEGIES:
        valid = ", ".join(sorted(_VALID_CUDA_ARCH_STRATEGIES))
        raise ValueError(f"Invalid cuda_arch_strategy {cuda_arch_strategy!r}; expected one of: {valid}")
    root = get_project_root()
    res = _build_cmake_args_from_env(cuda_arch_strategy) + _build_cmake_args_package_scm_version(root)
    return res


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        valid = ", ".join(sorted(_VALID_CUDA_ARCH_STRATEGIES))
        raise SystemExit(f"usage: {sys.argv[0]} <{valid}>")

    # Print arguments one per line for easy consumption in bash arrays
    for a in build_cmake_args(sys.argv[1]):
        print(a)
