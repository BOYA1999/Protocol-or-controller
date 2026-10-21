import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

import core

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/derived/coincident_d10_dispatch.npz")
    output = parser.parse_args().output.resolve()
    source_manifest = ROOT / "data/citylearn2022/manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    names = {"schema.json", "pricing.csv", "carbon_intensity.csv"}
    names.update(f"Building_{i}.csv" for i in range(1, 18))
    inputs = {}
    for item in source["files"]:
        if item["file"] in names:
            path = source_manifest.parent / item["file"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"source hash mismatch: {item['file']}")
            inputs[path.relative_to(ROOT).as_posix()] = digest
    if len(inputs) != len(names):
        raise ValueError("source manifest does not cover all dispatch inputs")
    data = core.load_main()
    buildings = data["buildings"]
    month = buildings[0]["month"]
    for b in buildings:
        np.testing.assert_array_equal(b["month"], month)
    arrays = dict(nob=np.stack([b["load"]-b["pv_kw"] for b in buildings]),
                  month=month, price=data["price"])
    for market in core.MARKETS:
        arrays[market] = np.stack([
            core.solve_oracle_demand(b["load"], b["pv_kw"], data["price"], data["carbon"],
                                     month, 6.4, 5.0, market, 10.0).net
            for b in buildings])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = dict(
        file=output.name, npz_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source=source["source"], source_revision=source["revision"],
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        source_files_sha256=inputs,
        code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ("code/core.py", "code/build_d10_dispatch.py")},
        design=dict(building_order=[b["id"] for b in buildings], n_hours=len(month),
                    capacity_kwh=6.4, power_kw=5.0, demand_rate_usd_per_kw_month=10.0,
                    markets=list(core.MARKETS), chronology="original continuous rows",
                    dispatch="independent per-building perfect-foresight LP; no district optimisation"),
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                          sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                for key, value in arrays.items()},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    output.with_name(output.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(n_lp=34, npz_sha256=manifest["npz_sha256"],
                         arrays={key: list(value.shape) for key, value in arrays.items()})))


if __name__ == "__main__":
    main()
