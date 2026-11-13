from __future__ import annotations

import random

SERVICES = ("auth-api", "payment-gateway", "worker-pool", "cache-node", "ingress-proxy", "scheduler")
REGIONS = ("eu-central-1", "eu-west-2", "us-east-1")

TRAIN_NORMAL_TEMPLATES = (
    "INFO service={svc} region={reg} status=200 latency_ms={lat} healthcheck passed",
    "INFO service={svc} region={reg} processed batch size={cnt} without errors",
    "INFO service={svc} region={reg} connection pool steady active={cnt} idle=4",
    "INFO service={svc} region={reg} cache hit ratio stable at 9{idx}%",
    "DEBUG service={svc} region={reg} heartbeat acknowledged seq={cnt}",
    "INFO service={svc} region={reg} scheduled cleanup completed in {lat}ms",
)

TRAIN_INCIDENT_TEMPLATES = (
    "ERROR service={svc} region={reg} status=503 upstream timeout after {lat}ms",
    "CRITICAL service={svc} region={reg} out of memory oom killer terminated worker",
    "ERROR service={svc} region={reg} database connection refused retries exhausted",
    "FATAL service={svc} region={reg} disk full write failure on volume /var/lib/data",
    "WARN service={svc} region={reg} sustained cpu saturation 99% queue backlog={cnt}",
    "ERROR service={svc} region={reg} unhandled exception traceback tls handshake failed",
)

TEST_NORMAL_TEMPLATES = (
    "INFO service={svc} region={reg} readiness probe ok response_time={lat}ms",
    "INFO service={svc} region={reg} replicated checkpoint seq={cnt} verified",
    "INFO service={svc} region={reg} routine token refresh completed count={cnt}",
)

TEST_INCIDENT_TEMPLATES = (
    "ERROR service={svc} region={reg} circuit breaker opened due to HTTP 502 downstream failure",
    "CRITICAL service={svc} region={reg} segmentation fault crash loop detected pid={cnt}",
    "ERROR service={svc} region={reg} authentication failed invalid signature burst count={cnt}",
)

# Challenging boundary cases included only in held-out evaluation to test keyword ambiguity
# and non-standard severity formatting without leaking into training templates.
CHALLENGE_NORMAL_TEMPLATES = (
    "INFO service={svc} region={reg} oom_guard initialized budget_ok free_mem=68%",
    "INFO service={svc} region={reg} timeout_budget=5000ms completed cleanly errors=0",
    "INFO service={svc} region={reg} exception_handler registered retries=0 status=200",
)

CHALLENGE_INCIDENT_TEMPLATES = (
    "WARN service={svc} region={reg} upstream connection reset by peer queue_depth={cnt}",
    "NOTICE service={svc} region={reg} kernel panic detected during disk volume flush",
    "service={svc} region={reg} unhandled fatal crash loop restarting pid={cnt}",
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
    """Generate deterministic, template-disjoint training and held-out evaluation splits."""
    rng = random.Random(seed)
    train_data = (
        _render_samples(TRAIN_NORMAL_TEMPLATES, 0, train_per_class, rng)
        + _render_samples(TRAIN_INCIDENT_TEMPLATES, 1, train_per_class, rng)
    )

    challenge_count = max(0, min(15, test_per_class // 8))
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
