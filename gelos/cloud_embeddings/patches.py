def resolve_patch_indices(spec: dict, n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    patches = spec["patches"]
    if isinstance(patches, str):
        prefix = "center_"
        if not patches.startswith(prefix) or "x" not in patches[len(prefix) :]:
            raise ValueError(
                f"patches string '{patches}' must match 'center_NxN' (e.g. 'center_2x2')"
            )
        n_str, _, m_str = patches[len(prefix) :].partition("x")
        if n_str != m_str:
            raise ValueError(
                f"patches string '{patches}' must be square (N == M); got {n_str}x{m_str}"
            )
        try:
            n = int(n_str)
        except ValueError as exc:
            raise ValueError(f"could not parse N from patches='{patches}'") from exc
        if n > min(n_rows, n_cols):
            raise ValueError(f"patches='{patches}' requires {n}x{n} but grid is {n_rows}x{n_cols}")
        r0 = (n_rows - n) // 2
        c0 = (n_cols - n) // 2
        return [(r0 + dr, c0 + dc) for dr in range(n) for dc in range(n)]

    resolved: list[tuple[int, int]] = []
    for pair in patches:
        if len(pair) != 2:
            raise ValueError(f"patch pair {pair} must be [row, col]")
        r, c = int(pair[0]), int(pair[1])
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise ValueError(f"patch ({r}, {c}) out of range for grid {n_rows}x{n_cols}")
        resolved.append((r, c))
    return resolved


def resolve_timestep_indices(spec: dict, T: int) -> list[int]:
    timesteps = spec["timesteps"]
    if isinstance(timesteps, str):
        if timesteps != "all":
            raise ValueError(f"timesteps string must be 'all', got '{timesteps}'")
        return list(range(T))
    resolved = [int(t) for t in timesteps]
    for t in resolved:
        if not (0 <= t < T):
            raise ValueError(f"timestep {t} out of range [0, {T})")
    return resolved
