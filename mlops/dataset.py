from __future__ import annotations

import random

SERVICES = ("auth-api", "payment-gateway", "worker-pool", "cache-node", "ingress-proxy", "scheduler")
REGIONS = ("eu-central-1", "eu-west-2", "us-east-1")

TRAIN_NORMAL_TEMPLATES = (
    "INFO service={svc} region={reg} status=200 latency_ms={lat} healthcheck passed errors=0",
    "INFO service={svc} region={reg} processed batch size={cnt} without errors status=200",
    "INFO service={svc} region={reg} connection pool steady active={cnt} idle=4 health=ok",
    "INFO service={svc} region={reg} cache hit ratio stable at 9{idx}% errors=0",
    "DEBUG service={svc} region={reg} heartbeat acknowledged seq={cnt} health=ok",
    "INFO service={svc} region={reg} scheduled cleanup completed in {lat}ms status=200",
    "WARN service={svc} region={reg} slow query notice latency_ms={lat} recovered=true errors=0",
    "service={svc} region={reg} status=200 duration_ms={lat} readiness check passed",
)

TRAIN_INCIDENT_TEMPLATES = (
    "ERROR service={svc} region={reg} status=503 upstream timeout after {lat}ms",
    "CRITICAL service={svc} region={reg} out of memory oom killer terminated worker",
    "ERROR service={svc} region={reg} database connection refused retries exhausted",
    "FATAL service={svc} region={reg} disk full write failure on volume /var/lib/data",
    "WARN service={svc} region={reg} sustained cpu saturation 99% queue backlog={cnt} timeout",
    "ERROR service={svc} region={reg} unhandled exception traceback tls handshake failed",
    "INFO service={svc} region={reg} status=502 retry=3 aborted connection reset by peer",
    "service={svc} region={reg} status=500 panic crash loop detected worker_id={cnt}",
)

TEST_NORMAL_TEMPLATES = (
    "INFO service={svc} region={reg} readiness check health=ok latency_ms={lat}ms errors=0",
    "INFO service={svc} region={reg} processed batch seq={cnt} completed without errors status=200",
    "DEBUG service={svc} region={reg} scheduled cleanup passed status=200 health=ok",
    "WARN service={svc} region={reg} slow query latency_ms={lat} recovered=true errors=0",
    "service={svc} region={reg} connection pool steady active={cnt} status=200 passed",
)

TEST_INCIDENT_TEMPLATES = (
    "ERROR service={svc} region={reg} upstream failure status=502 timeout after {lat}ms",
    "CRITICAL service={svc} region={reg} worker crash loop terminated oom detected",
    "ERROR service={svc} region={reg} tls handshake failed connection refused retries exhausted",
    "INFO service={svc} region={reg} status=503 aborted connection reset upstream backlog={cnt}",
    "service={svc} region={reg} status=500 unhandled exception panic write failure worker_id={cnt}",
)

# Boundary / unprefixed cases kept strictly out of training to prevent first-token shortcuts.
CHALLENGE_NORMAL_TEMPLATES = (
    "INFO service={svc} region={reg} oom_guard initialized budget_ok free_mem=68%",
    "INFO service={svc} region={reg} timeout_budget=5000ms completed cleanly errors=0",
    "WARN service={svc} region={reg} optional feature flag disabled by operator",
    "service={svc} region={reg} background compaction finished freed_mb={cnt} status=ok",
)

CHALLENGE_INCIDENT_TEMPLATES = (
    "WARN service={svc} region={reg} upstream connection reset by peer queue_depth={cnt}",
    "NOTICE service={svc} region={reg} kernel panic detected during disk volume flush",
    "service={svc} region={reg} unhandled fatal crash loop restarting pid={cnt}",
    "INFO service={svc} region={reg} deadlock detected waiting on shard_lock={cnt}",
)


def _render_samples(templates: tuple[str, ...], label: int, count: int, rng: random.Random) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for i in range(count):
        tpl = templates[i % len(templates)]
        msg = tpl.format(
            svc=rng.choice(SERVICES),
            reg=rng.choice(REGIONS),
            lat=rng.randint(12, 180) if label == 0 else rng.randint(1500, 9500),
            cnt=rng.randint(10, 450),
            idx=rng.randint(1, 9),
        )
        samples.append((msg, label))
    return samples


def build_disjoint_dataset(
    seed: int = 42,
    train_per_class: int = 360,
    test_per_class: int = 120,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    train_data = (
        _render_samples(TRAIN_NORMAL_TEMPLATES, 0, train_per_class, rng)
        + _render_samples(TRAIN_INCIDENT_TEMPLATES, 1, train_per_class, rng)
    )
    challenge_count = max(0, min(12, test_per_class // 10))
    standard_count = test_per_class - challenge_count
    test_data = (
        _render_samples(TEST_NORMAL_TEMPLATES, 0, standard_count, rng)
        + _render_samples(CHALLENGE_NORMAL_TEMPLATES, 0, challenge_count, rng)
        + _render_samples(TEST_INCIDENT_TEMPLATES, 1, standard_count, rng)
        + _render_samples(CHALLENGE_INCIDENT_TEMPLATES, 1, challenge_count, rng)
    )
    rng.shuffle(train_data)
    rng.shuffle(test_data)
    return train_data, test_data
