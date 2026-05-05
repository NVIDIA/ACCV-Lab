from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring

ATTRIBUTE_DIRECTIVE_RE = re.compile(
    r"^\s*\.\.\s+attribute::\s+([A-Za-z_]\w*)\s*$",
    re.MULTILINE,
)
IVAR_RE = re.compile(r"^\s*:ivar\s+([A-Za-z_]\w*):", re.MULTILINE)

# Cache per class and Napoleon setting combination because those settings change
# how "Attributes:" sections are converted into Sphinx attribute directives.
_DOCUMENTED_ATTRIBUTE_CACHE: dict[tuple[type[Any], bool, bool, bool], set[str]] = {}


def _extract_documented_attributes(docstring: str) -> set[str]:
    return set(ATTRIBUTE_DIRECTIVE_RE.findall(docstring)) | set(IVAR_RE.findall(docstring))


def _resolve_current_autodoc_class(app: Sphinx) -> type[Any] | None:
    env = getattr(app, "env", None)
    if env is None:
        return None

    module_name = env.temp_data.get("autodoc:module")
    class_path = env.temp_data.get("autodoc:class")
    if not module_name or not class_path:
        return None

    try:
        current_obj: Any = importlib.import_module(module_name)
        for part in class_path.split("."):
            current_obj = getattr(current_obj, part)
    except Exception:
        return None

    return current_obj if isinstance(current_obj, type) else None


def _documented_class_attributes(app: Sphinx, cls: type[Any]) -> set[str]:
    cache_key = (
        cls,
        bool(app.config.napoleon_google_docstring),
        bool(app.config.napoleon_numpy_docstring),
        bool(app.config.napoleon_use_ivar),
    )
    if cache_key in _DOCUMENTED_ATTRIBUTE_CACHE:
        return _DOCUMENTED_ATTRIBUTE_CACHE[cache_key]

    doc = inspect.getdoc(cls) or ""
    documented = _extract_documented_attributes(doc)

    # Re-run the class docstring through Napoleon so Google/NumPy style
    # "Attributes:" sections can be detected before skipping NamedTuple fields.
    parser_args = {
        "config": app.config,
        "app": app,
        "what": "class",
        "name": cls.__qualname__,
        "obj": cls,
    }
    if app.config.napoleon_google_docstring:
        documented.update(_extract_documented_attributes(str(GoogleDocstring(doc, **parser_args))))
    if app.config.napoleon_numpy_docstring:
        documented.update(_extract_documented_attributes(str(NumpyDocstring(doc, **parser_args))))

    _DOCUMENTED_ATTRIBUTE_CACHE[cache_key] = documented
    return documented


def skip_namedtuple_field_if_docstring_covers_it(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: dict[str, bool],
) -> bool | None:
    del obj, skip, options

    # autodoc calls this hook for every member. Returning None lets Sphinx and
    # other extensions keep their default decision for anything outside classes.
    if what != "class":
        return None

    current_class = _resolve_current_autodoc_class(app)
    if current_class is None:
        return None

    # NamedTuple classes expose fields through _fields. We only want to hide
    # those field descriptors, not ordinary class attributes or methods.
    fields = getattr(current_class, "_fields", None)
    if not (issubclass(current_class, tuple) and isinstance(fields, tuple)):
        return None

    if name not in fields:
        return None

    documented_attributes = _documented_class_attributes(app, current_class)
    if name in documented_attributes:
        return True

    return None


def setup(app: Sphinx):
    # Register with Sphinx autodoc's member-skip event so the extension can
    # suppress duplicate NamedTuple field docs after autodoc has found them.
    app.connect(
        "autodoc-skip-member",
        skip_namedtuple_field_if_docstring_covers_it,
        # Higher priority than Sphinx's default of 500 so this runs later and
        # only applies the NamedTuple-specific skip after other listeners.
        priority=2000,
    )
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
