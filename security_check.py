"""
security_check.py
-----------------
Startup guard + manual probe for the October 2026 advisories (eMudhra CA
intermediate-key compromise; Brotli decoder CVE-2026-111497).

Why a runtime check on top of requirements.txt: Render reuses a cached
virtualenv between builds. pip only upgrades packages that are REQUIRED, so
a stale cached certifi/Brotli can survive a deploy. This makes the process
fail closed (certifi) or loudly (Brotli) instead of silently trusting the
compromised CAs.

Manual probe (run in a Render Shell, uses the same HTTP client as the bot):
    python security_check.py --probe
Both test hosts below MUST fail certificate verification.
"""

import logging
import sys
from importlib import metadata

logger = logging.getLogger("security_check")

MIN_CERTIFI = (2026, 10, 19)          # Render advisory: certifi>=2026.10.19
MIN_BROTLI = (1, 2, 1)                # PYSEC-2026-4322
MIN_BROTLICFFI = (1, 2, 1, 0)         # PYSEC-2026-4323

PROBE_URLS = (
    "https://dv-g3b-emudhra.we-have-a-problem.com",
    "https://dv-g3b-aaa-emudhra.we-have-a-problem.com",
)


def _parse(version: str) -> tuple:
    nums = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    return tuple(nums)


def _installed(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_environment() -> None:
    """Raises RuntimeError if certifi is too old; logs CRITICAL for a
    vulnerable Brotli decoder (not fatal: it only matters if a hostile
    server sends Content-Encoding: br)."""
    certifi_v = _installed("certifi")
    if certifi_v is None or _parse(certifi_v) < MIN_CERTIFI:
        raise RuntimeError(
            f"certifi {certifi_v or '(missing)'} is older than "
            f"{'.'.join(map(str, MIN_CERTIFI))}: its CA bundle still trusts the "
            "compromised eMudhra chain. Redeploy with 'Clear build cache & deploy'."
        )
    logger.info("TLS trust store OK: certifi %s (>= %s)", certifi_v, ".".join(map(str, MIN_CERTIFI)))

    for name, minimum in (("Brotli", MIN_BROTLI), ("brotlicffi", MIN_BROTLICFFI)):
        v = _installed(name)
        if v is not None and _parse(v) < minimum:
            logger.critical(
                "%s %s is vulnerable to CVE-2026-111497 (fixed in %s). "
                "Redeploy with 'Clear build cache & deploy'.",
                name, v, ".".join(map(str, minimum)),
            )


def probe() -> int:
    """Returns 0 if every test host is REJECTED, 1 otherwise."""
    import requests
    bad = 0
    for url in PROBE_URLS:
        try:
            requests.get(url, timeout=10)
        except requests.exceptions.SSLError:
            print(f"OK   rejected (certificate verification failed): {url}")
        except requests.exceptions.RequestException as exc:
            print(f"??   could not connect ({type(exc).__name__}), inconclusive: {url}")
            bad = bad or 2
        else:
            print(f"FAIL accepted a compromised chain: {url}")
            bad = 1
    return 1 if bad == 1 else (2 if bad == 2 else 0)


if __name__ == "__main__":
    if "--probe" in sys.argv:
        sys.exit(probe())
    check_environment()
