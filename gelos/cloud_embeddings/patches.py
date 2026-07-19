def resolve_patch_indices(spec: dict, n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    patches = spec["patches"]
    if isinstance(patches, str):
        prefix = "center_"
        if not patches.startswith(prefix) or "x" not in patches[len(prefix) :]:
            raise ValueError(
                f"patches string '{patches}' must match 'center_RxC' (e.g. 'center_1x4')"
            )
        r_str, _, c_str = patches[len(prefix) :].partition("x")
        try:
            R = int(r_str)
            C = int(c_str)
        except ValueError as exc:
            raise ValueError(f"could not parse RxC from patches='{patches}'") from exc
        if R > n_rows or C > n_cols:
            raise ValueError(f"patches='{patches}' requires {R}x{C} but grid is {n_rows}x{n_cols}")
        r0 = (n_rows - R) // 2
        c0 = (n_cols - C) // 2
        return [(r0 + dr, c0 + dc) for dr in range(R) for dc in range(C)]

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
