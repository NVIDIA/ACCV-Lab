---
name: docs-compliance
description: ACCV-Lab documentation conventions and pre-PR compliance check. INVOKE when creating or editing any .md or .rst file under docs/ or packages/*/docs/, when modifying a Python module/class/function/method docstring, or before opening a PR that touches documentation. Provides hard rules for Sphinx role usage, docstring formatting, public-API export requirements, admonition syntax, and a pre-PR checklist with verification commands.
---

# ACCV-Lab Documentation Compliance

## When this skill applies

- Creating or editing any `.md` / `.rst` file under `docs/` or `packages/*/docs/`
- Modifying a Python module, class, function, or method docstring (anything autodoc renders)
- Preparing a PR that touches documentation, samples, or public-API docstrings

## Authoritative project references

Read these for ground truth before deviating from any rule below:

- `docs/conf.py` — Sphinx configuration (extensions, autodoc options, custom handlers)
- `docs/guides/DOCUMENTATION_SETUP_GUIDE.md` — build pipeline & directory structure
- `docs/guides/FORMATTING_GUIDE.md` — Python/C++ formatting (also affects docstring rendering)
- `docs/spelling_wordlist.txt` — accepted technical-term whitelist
- `docs/_ext/` — local Sphinx extensions (`note_literalinclude`, `module_docstring`, `markdown_note_admonitions`)

## Hard rules

### Rule 1 — API references: use Sphinx roles, never bare backticks

For any `accvlab.*` symbol mentioned in narrative text, use the appropriate role so it cross-links in the rendered HTML.

| Symbol kind | MyST role (`.md`) | RST role (`.rst`) |
|---|---|---|
| Class | `` {py:class}`~accvlab.<pkg>.<Class>` `` | `` :class:`~accvlab.<pkg>.<Class>` `` |
| Method | `` {py:meth}`~accvlab.<pkg>.<Class>.<method>` `` | `` :meth:`~accvlab.<pkg>.<Class>.<method>` `` |
| Module function | `` {py:func}`~accvlab.<pkg>.<func>` `` | `` :func:`~accvlab.<pkg>.<func>` `` |
| Attribute | `` {py:attr}`~accvlab.<pkg>.<Class>.<attr>` `` | `` :attr:`~accvlab.<pkg>.<Class>.<attr>` `` |

```
Bad:
  See `lookup()` and `put()` for details.

Good:
  See {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.lookup`
  and  {py:meth}`~accvlab.on_demand_video_decoder.SharedGopStore.put`
  for details.
```

**Exclusions** — keep bare backticks for:
- Stdlib types (`RuntimeWarning`, `NamedTuple`, `multiprocessing.Lock`) — this project does not cross-ref stdlib in user docs
- Parameter names, field names, prose terms (`access_tick`, `flock`, `spawn`)
- API names appearing inside fenced code blocks (` ```python ` … ` ``` `) — only narrative prose gets roles

### Rule 2 — `Returns:` block formatting gotcha

In Google/NumPy-style docstrings, the **first line** after `Returns:` is silently parsed as a return-type annotation if it ends with `:`, even when a real type annotation is on the signature. This produces malformed return docs that look fine in the source but break in the rendered API table.

```
Bad:
    Returns:
        Tuple of three things:
            - first
            - second
            - third

Good:
    Returns:
        Tuple containing

        - first
        - second
        - third
```

Lead with prose that does **not** end in `:`, then a blank line, then the bullets.

### Rule 3 — Public API must be exported

A new public class or function will not appear in the auto-generated `api.rst` unless **both** of these hold:

1. It is imported in `packages/<pkg>/accvlab/<pkg>/__init__.py`
2. It is listed in that file's `__all__`

Internal helpers belong under `_internal/` and are not exported.

### Rule 4 — Type annotations on public APIs

Every public function parameter and return value must have a type annotation. `sphinx_autodoc_typehints` renders them into the docs; missing annotations produce gaps in the rendered API table.

```python
Good:
    def get_batch(self, refs: List[GopRef]) -> List[np.ndarray]:
        ...
```

### Rule 5 — Annotation must match docstring

When changing a function's signature (parameter types, return type, parameter names), update the corresponding `Args:` / `Returns:` lines in the docstring **in the same edit**. Stale docstrings vs. live signatures are caught in review.

### Rule 6 — No implementation details in user-facing docs

User-facing docs (`docs/`, `packages/*/docs/`, public-class docstrings) describe **what the user does**, not **how the framework is implemented**.

```
Bad (jargon / impl detail leaked to user):
  - put() acquires an flock for atomicity (double-check after acquiring the lock)
  - Returns the original decoder
  - Uses C++ GetGOP under the hood

Good:
  - put() acquires an flock for atomicity
  - Returns the underlying PyNvGopDecoder
  - Returns cached data without re-demuxing
```

If a phrase would prompt the question *"is there something the user should do?"*, rewrite it. Implementation notes belong in source-level comments or developer-facing docstrings under `_internal/`, not user docs.

### Rule 7 — Doc build must be warning-free

`./scripts/build_docs.sh` warnings and errors are **blocking**. Before requesting review:

```bash
./scripts/build_docs.sh 2>&1 | tee /tmp/docs_build.log
grep -iE 'warning|error' /tmp/docs_build.log
```

Resolve every new warning. Common sources: bad role syntax, missing `__all__` exports, malformed `Returns:` blocks, broken cross-refs, unknown spelling.

### Rule 8 — Admonitions: blockquote form for dual-readable files

Files that must render correctly in **both** GitHub/IDE preview **and** Sphinx HTML use the blockquote admonition pattern:

```md
> **ℹ️ Note**: Short tip for the reader.

> **⚠️ Important**: Crucial warning users must not miss.
```

The local `markdown_note_admonitions` extension converts these to Sphinx admonitions at build time. Multi-line notes are supported as long as every line starts with `>`.

Use fenced admonitions ```` ```{note} ```` / ```` ```{important} ```` **only** in files that are exclusively part of the built docs and never opened in GitHub/IDE.

### Rule 9 — Edit source, not mirror

Source-of-truth lives at `packages/<pkg>/docs/`. Files under `docs/contained_package_docs_mirror/<pkg>/docs/` are symlinks regenerated by `mirror_referenced_dirs.py` at build time. Editing the mirror is unreliable:

- **Best case:** the edit happens to land on the original file via the symlink — it works *accidentally*, not by design.
- **Worst case:** the next build regenerates the mirror entry as a fresh file/symlink, **silently overwriting your edits**.

Always edit the source under `packages/<pkg>/docs/`.

### Rule 10 — Relative paths in include/image/literalinclude

Paths inside `.md` / `.rst` directives (`include`, `image`, `literalinclude`, etc.) must be **relative to the document that contains the directive**. Both absolute filesystem paths (`/home/...`) and repo-root-anchored absolute paths (`/packages/...`) will break — on other workstations and after the docs mirror step.

**For directives inside `.md` / `.rst` files (`docs/...` or `packages/<pkg>/docs/...`):**
paths are resolved relative to that `.md` / `.rst` file. Straightforward.

**For directives inside Python docstrings rendered by autodoc:** paths are resolved relative to the **docs file invoking the autodoc directive**, not the `.py` source file. To write a correct path:

1. Find the docs file that renders the docstring — usually `packages/<pkg>/docs/api.rst` (or another `api*.rst`).
2. Resolve every directive path from **that** docs file's location, not from the Python source file's location.
3. Prefer assets that already live under `packages/<pkg>/docs/`, referenced with a path from `docs/`, e.g. `.. image:: images/foo.png`.
4. For sibling directories at the package root (e.g. `samples/`, `examples/`) that are listed in `packages/<pkg>/docu_referenced_dirs.txt`, write the path **from `docs/`**, e.g. `.. literalinclude:: ../samples/demo.py`. The build step (`mirror_referenced_dirs.py`) ensures these sibling dirs are mirrored alongside `docs/` into `docs/contained_package_docs_mirror/<pkg>/`, so the same relative path resolves in both locations.

If the asset lives at the package root but is **not** listed in `docu_referenced_dirs.txt`, the path will resolve in the source tree but break after mirroring — add the directory to that file or move the asset under `docs/`.

```
Bad (machine-local absolute path — breaks for everyone but you):
  .. literalinclude:: /home/user/project/packages/foo/examples/demo.py

Bad (repo-root absolute path — still broken on other checkouts and in the mirror):
  .. literalinclude:: /packages/foo/examples/demo.py

Good (relative to packages/foo/docs/api.rst, with `examples` listed in docu_referenced_dirs.txt):
  .. literalinclude:: ../examples/demo.py
```

### Rule 11 — Sample docs: explain real-use-case provenance and cross-link

When a sample uses hard-coded values that would normally come from runtime sources (parser output, demuxer results, model outputs, etc.), explicitly document where those values come from in production. Cross-link to a related sample that demonstrates the real flow.

```python
Good:
    # Each task tuple: (video_path, target_frame_id, gop_first_frame, gop_len).
    #
    # In a real pipeline, gop_first_frame and gop_len would come from a
    # demuxer (e.g. GetGOPList returning first_frame_ids / gop_lens).
    # See samples/SampleSeparationAccessGOPListAPI.py for an end-to-end
    # example. Hard-coded values here keep the demo dependency-free.
    tasks = [...]
```

## Pre-PR compliance checklist

Run through these before requesting review on any PR that touches docs or docstrings:

- [ ] All `accvlab.*` API references in narrative use `{py:meth}` / `{py:func}` / `{py:class}` roles
- [ ] No bare backticks for `accvlab.*` names except in code blocks
- [ ] Admonitions in dual-readable files use the blockquote pattern
- [ ] Edits land in `packages/<pkg>/docs/`, not in the mirror
- [ ] Paths in directives are relative
- [ ] New technical terms added to `docs/spelling_wordlist.txt`
- [ ] `./scripts/build_docs.sh` runs with no new warnings or errors
- [ ] `./scripts/build_docs.sh --spelling` reviewed; report at `docs/_build/spelling/output.txt`
- [ ] Sample docs reference real-use-case origin and cross-link to related samples

## Verification commands

Quick scans to surface common violations before review:

```bash
# 1. Bare backtick API references that should be sphinx roles.
#    Customise the regex with the symbols touched by your PR.
grep -rnE '`(SharedGopStore|GopRef|CachedGopDecoder|PyNvGopDecoder|CreateGopDecoder)[A-Za-z_]*\(?\)?`' \
    docs/ packages/*/docs/ 2>/dev/null | grep -v '```'

# 2. Returns: block immediately followed by a line that ends in ':' (Rule 2 violation).
grep -rEn -A1 'Returns:$' packages/*/accvlab/ | grep -E ':\s*$'

# 3. Public symbols not exported in root __init__.py (manual diff).
#    After adding `class Foo` or `def bar`, confirm:
#      - `from .<sub> import Foo` (or `bar`) appears in __init__.py
#      - `'Foo'` (or `'bar'`) appears in __all__
grep -E '^(class|def) [A-Z]' packages/<pkg>/accvlab/<pkg>/<file>.py
grep -E "(<symbol>|__all__)" packages/<pkg>/accvlab/<pkg>/__init__.py

# 4. Accidental edits inside the mirror directory (Rule 9 violation).
git diff --name-only | grep contained_package_docs_mirror

# 5. Full doc build with warning surface.
./scripts/build_docs.sh 2>&1 | grep -iE 'warning|error' | grep -v -i 'INFO'

# 6. Spelling check.
./scripts/build_docs.sh --spelling
cat docs/_build/spelling/output.txt 2>/dev/null
```