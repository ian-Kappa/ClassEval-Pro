"""Stage 1 - GitHub mining.

Query the GitHub Search API for post-cutoff Python repos across the
11-domain keyword matrix, with stratified star-tier sampling. Dedup by
repo URL. Output: config.REPOS_FILE.
"""

import json
import time

import requests

from . import config


def _query(keywords, stars_range):
    """One GitHub repo search; returns items or []."""
    q = (f"language:python created:>={config.START_DATE} "
         f"stars:{stars_range} size:50..100000 ({' OR '.join(keywords)})")
    headers = {"Authorization": f"token {config.GITHUB_TOKEN}",
               "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get("https://api.github.com/search/repositories",
                          headers=headers,
                          params={"q": q, "sort": "stars", "order": "desc",
                                  "per_page": 30},
                          timeout=15)
        if r.status_code == 403:        # rate limited
            time.sleep(60)
            return []
        if r.status_code != 200:        # 422 = query too complex, skip
            return []
        return r.json().get("items", [])
    except requests.RequestException:
        return []


def run():
    if not config.GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN env var is required for stage 1")

    repos, seen = [], set()
    for domain, kws in config.DOMAIN_KEYWORDS.items():
        batches = [kws[i:i + config.KEYWORD_CHUNK]
                   for i in range(0, len(kws), config.KEYWORD_CHUNK)]
        for batch in batches:
            for strat in config.SEARCH_STRATEGIES:
                for item in _query(batch, strat["stars_range"]):
                    url = item["html_url"]
                    if url in seen:
                        continue
                    seen.add(url)
                    repos.append({
                        "category": domain,
                        "name": item["full_name"],
                        "url": url,
                        "stars": item["stargazers_count"],
                    })
                time.sleep(2)           # be gentle with the search API

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    config.REPOS_FILE.write_text(
        json.dumps(repos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage1] mined {len(repos)} repos -> {config.REPOS_FILE.name}")
    return len(repos)


if __name__ == "__main__":
    run()
