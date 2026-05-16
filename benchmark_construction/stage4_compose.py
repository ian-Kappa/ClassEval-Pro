"""Stage 4 - intra/cross-domain composition (paper Sec. 3.2, Table 2).

Render each structured class into an AST-augmented skeleton (signatures,
contracts, intra-class call dependencies, target functionalities, interface
constraints), then pair classes into INTRA_DOMAIN_TASKS same-domain and the
rest cross-domain combinations. Deterministic via config.SEED.
Output: config.PAIRS_FILE (records with skeleton1 / skeleton2).
"""

import ast
import json
import random
import re

from . import config


def _intra_class_calls(code: str, names: set) -> list:
    """Methods of the same class invoked from this method body."""
    if not code:
        return []
    found = set()
    try:
        for n in ast.walk(ast.parse(code)):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in ("self", "cls")
                    and n.func.attr in names):
                found.add(n.func.attr)
    except SyntaxError:
        for m in re.finditer(r"\b(?:self|cls)\.(\w+)\s*\(", code):
            if m.group(1) in names:
                found.add(m.group(1))
    return sorted(found)


def render_skeleton(sk: dict, domain: str) -> str:
    """Structured class JSON -> AST-augmented skeleton text (GAP 2 + GAP 3)."""
    methods = sk.get("methods") or []
    names = {m.get("name", "") for m in methods}
    meta = sk.get("metadata") or {}
    L = [f"# Class: {sk.get('class_name', '?')}  (Domain: {domain})",
         f"# Source: {meta.get('REPO', '?')} | LOC: {meta.get('LOC', '?')} "
         f"| Methods: {sk.get('stats', {}).get('total_methods', len(methods))}",
         "", "# Purpose",
         (sk.get("class_description") or "(no class docstring)").strip(),
         "", "# Imports (standard library only)"]
    L += sk.get("imports") or ["(none)"]
    L += ["", "# Interface (AST-extracted signatures & contracts)"]
    intents = []
    for m in methods:
        name = m.get("name", "")
        vis = ("dunder" if name.startswith("__") and name.endswith("__")
               else "private" if m.get("is_private") else "public")
        L.append(f"## {name}")
        L.append(f"- Signature: {(m.get('signature') or '').strip()}")
        L.append(f"- Params: {', '.join(m.get('params') or []) or '(none)'}")
        L.append(f"- Visibility: {vis}")
        calls = _intra_class_calls(m.get("code", ""), names)
        if calls:
            L.append(f"- Calls (intra-class deps): {', '.join(calls)}")
        doc = (m.get("docstring") or "").strip()
        if doc:
            head = doc.splitlines()[0].strip()
            L.append(f"- Doc: {head}")
            if vis == "public" and name != "__init__":
                intents.append(f"{name}: {head}")
        L.append("")
    L.append("# Target Functionalities")
    L += [f"- {x}" for x in intents] or [
        f"- {m['name']}" for m in methods if not m["name"].startswith("_")]
    L += ["", "# Interface Constraints",
          "- Preserve every method signature and parameter name above.",
          "- Use the Python standard library only.",
          "- Maintain documented return types and observable behavior.",
          "- Respect the intra-class call dependencies when merging logic."]
    return "\n".join(L)


def _domain_pair_allowed(d1: str, d2: str) -> bool:
    """Expert composable-domain matrix hook (None = any distinct pair)."""
    if config.COMPOSABLE_DOMAIN_PAIRS is None:
        return True
    allowed = {frozenset(p) for p in config.COMPOSABLE_DOMAIN_PAIRS}
    return frozenset((d1, d2)) in allowed


def _pair(classes: list) -> list:
    rng = random.Random(config.SEED)
    by_dom: dict = {}
    for c in classes:
        by_dom.setdefault(c["category"], []).append(c)
    use = {id(c): 0 for c in classes}
    free = lambda c: use[id(c)] < config.MAX_CLASS_REUSE
    pairs, seen = [], set()

    def take(a, b, t):
        k = frozenset((id(a), id(b)))
        if a is b or k in seen or not (free(a) and free(b)):
            return
        pairs.append((a, b, t))
        seen.add(k)
        use[id(a)] += 1
        use[id(b)] += 1

    # Intra-domain
    doms = [d for d, v in by_dom.items() if len(v) >= 2]
    for _ in range(config.INTRA_DOMAIN_TASKS * 200):
        if sum(p[2] == "intra" for p in pairs) >= config.INTRA_DOMAIN_TASKS \
                or not doms:
            break
        pool = [c for c in by_dom[rng.choice(doms)] if free(c)]
        if len(pool) >= 2:
            take(*rng.sample(pool, 2), "intra")

    # Cross-domain
    all_d = list(by_dom)
    cross_target = config.TOTAL_TASKS - config.INTRA_DOMAIN_TASKS
    for _ in range(cross_target * 400):
        if sum(p[2] == "cross" for p in pairs) >= cross_target \
                or len(all_d) < 2:
            break
        d1, d2 = rng.sample(all_d, 2)
        if not _domain_pair_allowed(d1, d2):
            continue
        la = [c for c in by_dom[d1] if free(c)]
        lb = [c for c in by_dom[d2] if free(c)]
        if la and lb:
            take(rng.choice(la), rng.choice(lb), "cross")

    rng.shuffle(pairs)
    return pairs


def run():
    items = json.loads(config.CLASSES_FILE.read_text(encoding="utf-8"))
    classes = [{"category": it["category"], "skeleton": it["skeleton"]}
               for it in items
               if isinstance(it.get("skeleton"), dict)
               and it["skeleton"].get("class_name")]
    if len(classes) < 2:
        raise RuntimeError("stage4 needs >= 2 source classes")

    pairs = _pair(classes)
    out = []
    for i, (a, b, t) in enumerate(pairs):
        out.append({
            "task_id": f"ClassEval_{i}",
            "composition_type": t,
            "domain1": a["category"], "domain2": b["category"],
            "ClassName1": a["skeleton"]["class_name"],
            "ClassName2": b["skeleton"]["class_name"],
            "skeleton1": render_skeleton(a["skeleton"], a["category"]),
            "skeleton2": render_skeleton(b["skeleton"], b["category"]),
        })
    config.PAIRS_FILE.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n_i = sum(p[2] == "intra" for p in pairs)
    print(f"[stage4] {len(out)} pairs (intra={n_i}, cross={len(out) - n_i}) "
          f"-> {config.PAIRS_FILE.name}")
    if len(out) < config.TOTAL_TASKS:
        print(f"[stage4] WARN: below TOTAL_TASKS={config.TOTAL_TASKS}; "
              f"raise MAX_CLASS_REUSE or mine more classes")
    return len(out)


if __name__ == "__main__":
    run()
