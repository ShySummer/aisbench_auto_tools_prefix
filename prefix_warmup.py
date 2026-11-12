"""Warm every prefix on every vLLM DP rank before measured traffic starts."""
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def validate_dp_ranks(dp, engine_ids):
    if dp < 1:
        raise ValueError("--dp must be positive")
    expected, actual = set(range(dp)), set(engine_ids)
    if actual != expected:
        raise ValueError(f"--dp {dp} requires engine ranks {sorted(expected)}, "
                         f"but metrics report {sorted(actual)}; use actual service DP size")


def chat_completions_url(host_ip, host_port, url=""):
    base = url.strip() if isinstance(url, str) else ""
    if base:
        if not base.startswith(("http://", "https://")):
            base = "http://" + base
    else:
        host = str(host_ip)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        base = f"http://{host}:{host_port}/"
    return urllib.parse.urljoin(base.rstrip("/") + "/", "v1/chat/completions")


def _load_prefixes(prefix_file, dp, prefix_num):
    if dp < 1 or prefix_num < 1:
        raise ValueError("--dp and --prefix_num must be positive")
    with open(prefix_file, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != dp * prefix_num:
        raise ValueError(f"Expected {dp * prefix_num} warmup rows, found {len(rows)}")
    prefixes = []
    for index in range(prefix_num):
        group = rows[index * dp:(index + 1) * dp]
        text = group[0].get("question")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Prefix {index} must contain a nonempty question")
        if any(row.get("question") != text for row in group):
            raise ValueError(f"Prefix {index} must have {dp} identical copies")
        prefixes.append(text)
    return prefixes


def _warm_rank(endpoint, model, api_key, text, rank, prefix_index,
               generation_kwargs, timeout):
    body = dict(generation_kwargs or {})
    body.update(model=model, messages=[{"role": "user", "content": text}],
                max_tokens=1, stream=False)
    headers = {"Content-Type": "application/json", "X-data-parallel-rank": str(rank)}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST")
    # Inference traffic must not inherit the proxy used to fetch GitHub.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Prefix {prefix_index}, DP{rank}: HTTP {exc.code}; warmup aborted") from None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Prefix {prefix_index}, DP{rank}: {type(exc).__name__}; warmup aborted") from None
    if not isinstance(result, dict) or result.get("error") or not result.get("choices"):
        raise RuntimeError(f"Prefix {prefix_index}, DP{rank}: invalid completion response")
    usage = result.get("usage") or {}
    record = {"rank": rank, "prefix_index": prefix_index,
              "elapsed_seconds": round(time.monotonic() - started, 4),
              "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens")}
    logging.info("[prefix warmup] prefix=%d DP%d completed, prompt_tokens=%s",
                 prefix_index, rank, record["prompt_tokens"])
    return record


def warmup_prefixes(prefix_file, dp, prefix_num, *, host_ip, host_port,
                    model, api_key="", url="", generation_kwargs=None,
                    timeout=600, report_path=None):
    """One request per (prefix, rank); return only after all complete.

    The endpoint must honor X-data-parallel-rank, as supported by vLLM.
    Any reverse proxy must preserve this header.
    """
    prefixes = _load_prefixes(prefix_file, dp, prefix_num)
    endpoint = chat_completions_url(host_ip, host_port, url)
    records = []
    with ThreadPoolExecutor(max_workers=dp) as executor:
        for index, text in enumerate(prefixes):
            futures = [executor.submit(
                _warm_rank, endpoint, model, api_key, text, rank, index,
                generation_kwargs, timeout) for rank in range(dp)]
            for future in as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda item: (item["prefix_index"], item["rank"]))
    report = {"dp_size": dp, "prefix_count": prefix_num,
              "completed_requests": len(records), "requests": records}
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def wait_for_prefix_metrics(report, initial_queries, read_queries, timeout=30):
    """Wait until async metrics include warmup, so measured deltas exclude it."""
    required = {rank: initial_queries.get(rank, 0) for rank in range(report['dp_size'])}
    for record in report['requests']:
        count = record.get('prompt_tokens')
        if not isinstance(count, int) or count < 1:
            raise RuntimeError('Warmup response has no valid prompt_tokens usage')
        required[record['rank']] += count
    deadline = time.monotonic() + timeout
    while True:
        current = read_queries()
        if all(current.get(rank, -1) >= count for rank, count in required.items()):
            return current
        if time.monotonic() >= deadline:
            raise RuntimeError('Warmup metrics did not catch up; measured test was not started')
        time.sleep(0.5)
