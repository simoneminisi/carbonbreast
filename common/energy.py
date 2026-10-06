"""CodeCarbon energy tracking (verbatim from pipeline.py / pipeline_keras.py) + trunk sidecars."""
import contextlib
import json
import os
import time


@contextlib.contextmanager
def track_energy(label, enabled=True):
    """Report energy (Wh) + CO2 (g) via CodeCarbon and wall-time (s, measured independently).
    Yields a dict filled on exit with keys wh / co2_g / seconds. Degrades to wall-time only if
    CodeCarbon is absent or errors (e.g. offline box), so a run never crashes."""
    out = {"wh": None, "co2_g": None, "seconds": None}
    tracker = None
    if enabled:
        try:
            from codecarbon import EmissionsTracker  # noqa: PLC0415 (optional dep)
            tracker = EmissionsTracker(log_level="error", save_to_file=False)
            tracker.start()
        except Exception as e:  # noqa: BLE001
            print(f"[energy] CodeCarbon unavailable ({type(e).__name__}: {e}); wall-time only")
    t0 = time.time()
    try:
        yield out
    finally:
        out["seconds"] = time.time() - t0
        if tracker is not None:
            try:
                co2_kg = tracker.stop()
                out["co2_g"] = (co2_kg or 0.0) * 1000.0
                out["wh"] = tracker._total_energy.kWh * 1000.0
            except Exception as e:  # noqa: BLE001
                print(f"[energy] CodeCarbon stop failed ({type(e).__name__}: {e}); wall-time only")
        wh = f"{out['wh']:.2f}" if out["wh"] is not None else "n/a"
        co2 = f"{out['co2_g']:.3f}" if out["co2_g"] is not None else "n/a"
        print(f"[energy] {label}: {wh} Wh  {co2} gCO2  {out['seconds']:.1f} s")


def energy_sidecar_path(path):
    """trunk_tf_x.npz -> trunk_tf_x.energy.json (also handles the keras .weights.h5 name)."""
    if path.endswith(".weights.h5"):
        return path[:-len(".weights.h5")] + ".energy.json"
    return os.path.splitext(path)[0] + ".energy.json"


def write_sidecar(trunk_path, backbone, val_top1, energy):
    sidecar = energy_sidecar_path(trunk_path)
    with open(sidecar, "w") as f:
        json.dump({"backbone": backbone, "val_top1": val_top1,
                   "wh": energy["wh"], "co2_g": energy["co2_g"], "seconds": energy["seconds"]}, f)
    print(f"wrote energy sidecar -> {sidecar}")
    return sidecar


def read_sidecar(trunk_path):
    """Stage-1 energy dict for a trunk, or None when no sidecar exists."""
    if not trunk_path:
        return None
    sidecar = energy_sidecar_path(trunk_path)
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        return json.load(f)
