"""Policy-invariant guest environment layout for gem5 SE workloads."""

TARGET_ENV_BYTES = 16384


def finalize_environment(entries):
    keys = [entry.split("=", 1)[0] for entry in entries]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate guest environment key")
    prefix = "GRAPHBREW_ENV_PAD="
    current = sum(len(entry.encode()) + 1 for entry in entries)
    padding = TARGET_ENV_BYTES - current - len(prefix.encode()) - 1
    if padding < 0:
        raise RuntimeError(
            f"guest environment exceeds {TARGET_ENV_BYTES} bytes")
    result = tuple(entries) + (prefix + ("0" * padding),)
    actual = sum(len(entry.encode()) + 1 for entry in result)
    if actual != TARGET_ENV_BYTES:
        raise RuntimeError(
            f"guest environment layout mismatch: {actual}")
    return result
