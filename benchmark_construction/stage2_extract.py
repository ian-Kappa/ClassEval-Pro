"""Stage 2 - extract, validate, structure.

For each mined repo: shallow-clone, AST-parse non-test files, keep classes
that (a) import only stdlib, (b) have >= MIN_METHODS methods, (c) are within
MIN_LOC..MAX_LOC. Then 3-level validation (content / compile / unresolved-
name) and serialization to a structured JSON record per class.
Output: config.STRUCTURED_FILE (a single JSON list).
"""

import ast
import builtins
import json
import shutil
import subprocess
import sys
import uuid
import warnings
from pathlib import Path

from . import config

_STD = set(getattr(sys, "stdlib_module_names", set())) | {
    "os", "sys", "io", "re", "json", "csv", "math", "random", "typing",
    "collections", "itertools", "functools", "datetime", "pathlib"}
_BUILTINS = set(dir(builtins))


def _safe_rmtree(path: Path):
    shutil.rmtree(path, ignore_errors=True)


class _FileAnalyzer(ast.NodeVisitor):
    """Collect stdlib-only classes meeting the structural constraints."""

    def __init__(self, src: str):
        self.src = src
        self.clean = True
        self.imports: list = []
        self.candidates: list = []

    def _check_module(self, base: str):
        if base not in _STD:
            self.clean = False

    def visit_Import(self, node):
        for a in node.names:
            self._check_module(a.name.split(".")[0])
        if self.clean:
            seg = ast.get_source_segment(self.src, node)
            if seg:
                self.imports.append(seg)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level > 0:                       # relative import -> reject
            self.clean = False
        elif node.module:
            self._check_module(node.module.split(".")[0])
        if self.clean:
            seg = ast.get_source_segment(self.src, node)
            if seg:
                self.imports.append(seg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if not self.clean:
            return
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if len(methods) < config.MIN_METHODS:
            return
        loc = node.end_lineno - node.lineno
        if not (config.MIN_LOC <= loc <= config.MAX_LOC):
            return
        body = ast.get_source_segment(self.src, node)
        code = "\n".join(sorted(set(self.imports))) + "\n\n" + body
        self.candidates.append(
            {"class_name": node.name, "loc": loc,
             "methods_count": len(methods), "code": code})


class _UnresolvedNameChecker(ast.NodeVisitor):
    """Scope-aware visitor flagging names used but never bound (self-contain)."""

    def __init__(self):
        self.globals = set(_BUILTINS)
        self.undefined = set()
        self.scopes = [set()]

    def _defined(self, name):
        return any(name in s for s in reversed(self.scopes)) \
            or name in self.globals

    def visit_Import(self, node):
        for a in node.names:
            self.globals.add(a.asname or a.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.globals.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.globals.add(node.name)
        self.scopes.append(set())
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node):
        (self.globals if len(self.scopes) == 1
         else self.scopes[-1]).add(node.name)
        scope = {a.arg for a in node.args.args}
        if node.args.vararg:
            scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            scope.add(node.args.kwarg.arg)
        self.scopes.append(scope)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.scopes[-1].add(node.id)
        elif isinstance(node.ctx, ast.Load) and not self._defined(node.id):
            self.undefined.add(node.id)


def _validate(code: str) -> bool:
    """3-level check: non-empty/no-markup, compiles, no unresolved names."""
    if not code.strip() or code.lstrip().startswith("<"):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(code, "<candidate>", "exec")
        tree = ast.parse(code)
    except Exception:
        return False
    chk = _UnresolvedNameChecker()
    chk.visit(tree)
    return not {n for n in chk.undefined if n != "self"}


def _structure(code: str, class_name: str) -> dict | None:
    """Serialize a validated class into a structured JSON record."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    imports, info = [], None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(code, node)
            if seg:
                imports.append(seg)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = []
            for it in node.body:
                if not isinstance(it, ast.FunctionDef):
                    continue
                try:
                    sig = f"def {it.name}({ast.unparse(it.args)})"
                    if it.returns:
                        sig += f" -> {ast.unparse(it.returns)}"
                    sig += ":"
                except Exception:
                    sig = f"def {it.name}(...):"
                methods.append({
                    "name": it.name,
                    "signature": sig,
                    "params": [a.arg for a in it.args.args if a.arg != "self"],
                    "docstring": ast.get_docstring(it) or "",
                    "code": ast.get_source_segment(code, it) or "",
                    "is_init": it.name == "__init__",
                    "is_private": it.name.startswith("_")
                    and not it.name.startswith("__"),
                })
            info = {"class_name": node.name,
                    "class_description": ast.get_docstring(node) or "",
                    "methods": methods}
            break
    if not info:
        return None
    pub = [m for m in info["methods"] if not m["name"].startswith("_")]
    return {
        "imports": sorted(set(imports)),
        "class_name": info["class_name"],
        "class_description": info["class_description"],
        "methods": info["methods"],
        "stats": {"total_methods": len(info["methods"]),
                  "public_methods": len(pub)},
    }


def _process_repo(repo: dict) -> list:
    tmp = config.ARTIFACTS / "_clones" / str(uuid.uuid4())
    out = []
    try:
        subprocess.run(["git", "clone", "--depth",
                        str(config.CLONE_DEPTH), repo["url"], str(tmp)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=config.CLONE_TIMEOUT, check=True)
        for py in tmp.rglob("*.py"):
            low = str(py).lower()
            if config.SKIP_TEST_FILES and ("test" in low or "migration" in low):
                continue
            try:
                content = py.read_text(encoding="utf-8", errors="ignore")
                if any(fw in content.lower()
                       for fw in ("django", "flask", "fastapi")):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree = ast.parse(content)
            except Exception:
                continue
            an = _FileAnalyzer(content)
            an.visit(tree)
            if not an.clean:
                continue
            for cand in an.candidates:
                if not _validate(cand["code"]):
                    continue
                rec = _structure(cand["code"], cand["class_name"])
                if not rec:
                    continue
                rec["metadata"] = {"REPO": repo["name"],
                                   "CATEGORY": repo["category"],
                                   "CLASS": cand["class_name"],
                                   "LOC": str(cand["loc"]),
                                   "STARS": repo.get("stars", 0)}
                out.append({"category": repo["category"], "skeleton": rec})
    except (subprocess.SubprocessError, OSError):
        pass
    finally:
        _safe_rmtree(tmp)
    return out


def run():
    repos = json.loads(config.REPOS_FILE.read_text(encoding="utf-8"))
    classes = []
    for i, repo in enumerate(repos, 1):
        got = _process_repo(repo)
        classes.extend(got)
        print(f"[stage2] {i}/{len(repos)} {repo['name']}: +{len(got)} classes")
    _safe_rmtree(config.ARTIFACTS / "_clones")
    config.STRUCTURED_FILE.write_text(
        json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage2] {len(classes)} validated classes "
          f"-> {config.STRUCTURED_FILE.name}")
    return len(classes)


if __name__ == "__main__":
    run()
