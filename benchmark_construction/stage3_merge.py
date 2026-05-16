"""Stage 3 - merge & The Stack-style deduplication.

Consolidate structured classes, then deduplicate following The Stack
methodology (paper Sec. 3.2): (a) cheap (repo, class) key dedup, (b) exact
content dedup via SHA-256 of the canonicalized class source, (c) near-
duplicate removal via MinHash-estimated Jaccard similarity. Finally sort by
domain, assign stable ids, and report dataset statistics.
Output: config.CLASSES_FILE.
"""

import hashlib
import json
import re

from . import config


def _canonical(sk: dict) -> str:
    """Whitespace-normalized class source used for content hashing."""
    parts = list(sk.get("imports") or [])
    parts += [m.get("code", "") for m in sk.get("methods") or []]
    text = "\n".join(parts)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _shingles(code: str) -> set:
    """Token n-gram set (k = config.SHINGLE_K)."""
    toks = re.findall(r"\w+|\S", code)
    k = config.SHINGLE_K
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _minhash(shingles: set) -> tuple:
    """MinHash signature over config.MINHASH_PERM hash permutations."""
    if not shingles:
        return tuple([0] * config.MINHASH_PERM)
    sig = []
    for seed in range(config.MINHASH_PERM):
        sig.append(min(int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16)
                        for s in shingles))
    return tuple(sig)


def _jaccard(a: tuple, b: tuple) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def run():
    items = json.loads(config.STRUCTURED_FILE.read_text(encoding="utf-8"))

    seen_key, seen_hash, sigs, merged = set(), set(), [], []
    for it in items:
        sk = it["skeleton"]
        meta = sk.get("metadata", {})

        # (a) cheap (repo, class) key dedup
        key = (meta.get("REPO", ""), sk.get("class_name", ""))
        if key in seen_key:
            continue

        # (b) exact content dedup
        canon = _canonical(sk)
        h = hashlib.sha256(canon.encode()).hexdigest()
        if h in seen_hash:
            continue

        # (c) MinHash near-duplicate dedup
        sig = _minhash(_shingles(canon))
        if any(_jaccard(sig, s) >= config.DEDUP_NEAR_JACCARD for s in sigs):
            continue

        seen_key.add(key)
        seen_hash.add(h)
        sigs.append(sig)
        merged.append(it)

    merged.sort(key=lambda x: x["category"])
    for i, it in enumerate(merged):
        it["task_id"] = i

    config.CLASSES_FILE.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    dropped = len(items) - len(merged)
    by_dom: dict = {}
    for it in merged:
        by_dom[it["category"]] = by_dom.get(it["category"], 0) + 1
    print(f"[stage3] {len(merged)} unique classes "
          f"({dropped} duplicates removed) across {len(by_dom)} domains "
          f"-> {config.CLASSES_FILE.name}")
    for d, n in sorted(by_dom.items()):
        print(f"         {d:<24} {n}")
    return len(merged)


if __name__ == "__main__":
    run()
