import os
import re
from pathlib import Path
from typing import List, Optional

# Marker file at the ACCV-Lab monorepo root (see `.nav` in the repository).
_NAV_MARKER = ".nav"
# Must match `.nav` contents after strip (UTF-8); see repository root `.nav`.
_NAV_EXPECTED_CONTENT = "project root"


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


def _detect_cuda_architectures() -> List[str]:
    """
    Try to detect CUDA architectures from PyTorch if available.
    Returns a list like ['70', '75', '80'] ([] if not detected).
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return []
        arches: List[str] = []
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arch = f"{major}{minor}"
            if arch not in arches:
                arches.append(arch)
        return arches
    except Exception:
        return []


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


def _build_cmake_args_from_env() -> List[str]:
    """
    Build a list of -D CMake arguments from environment variables to harmonize
    build configuration across setuptools, external CMake, and scikit-build flows.
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

    # CUSTOM_CUDA_ARCHS -> CMAKE_CUDA_ARCHITECTURES
    custom_archs = os.environ.get("CUSTOM_CUDA_ARCHS")
    if custom_archs:
        # Accept comma or semicolon separated
        norm_archs = custom_archs.replace(",", ";")
        args.append(f'-DCMAKE_CUDA_ARCHITECTURES={norm_archs}')
    else:
        # Attempt auto-detection via torch; if empty, let CMake defaults apply
        detected = _detect_cuda_architectures()
        if detected:
            args.append(f'-DCMAKE_CUDA_ARCHITECTURES={";".join(detected)}')

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
    from setuptools_scm import get_version

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


def build_cmake_args() -> List[str]:
    """
    Full CMake -D list: environment-based flags plus repo-aligned SCM version define.
    """
    root = get_project_root()
    return _build_cmake_args_from_env() + _build_cmake_args_package_scm_version(root)


if __name__ == "__main__":
    # Print arguments one per line for easy consumption in bash arrays
    for a in build_cmake_args():
        print(a)
