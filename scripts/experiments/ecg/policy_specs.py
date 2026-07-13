"""Canonical ECG experiment policy parsing and output labels."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy: str
    ecg_mode: str | None = None
    charge_popt_overhead: bool = False
    ecg_schedule_k: int = 0
    ecg_stream_bypass: bool = False
    ecg_variant: str | None = None
    ecg_transport_pinned: bool = False

    @property
    def safe_label(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.label)


def parse_policy_spec(text: str) -> PolicySpec:
    upper = text.strip().upper().replace("-", "_")
    charge_popt = False
    explicit_charge = False
    if upper.endswith("_CHARGED"):
        upper = upper[: -len("_CHARGED")]
        charge_popt = True
        explicit_charge = True
    elif upper.endswith(":CHARGED"):
        upper = upper[: -len(":CHARGED")]
        charge_popt = True
        explicit_charge = True
    elif upper.endswith("_UNCHARGED"):
        upper = upper[: -len("_UNCHARGED")]
        explicit_charge = True
    elif upper.endswith(":UNCHARGED"):
        upper = upper[: -len(":UNCHARGED")]
        explicit_charge = True

    if upper in ("ECG:K2", "ECG_K2"):
        return PolicySpec(
            label="ECG_K2",
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_variant="adaptive",
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:K2_STREAMSHIELD",
        "ECG_K2_STREAMSHIELD",
        "ECG:K2_SS",
        "ECG_K2_SS",
    ):
        return PolicySpec(
            label="ECG_K2_STREAMSHIELD",
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_variant="adaptive",
            ecg_transport_pinned=True,
        )
    if upper in ("ECG:K1", "ECG_K1"):
        return PolicySpec(
            label="ECG_K1",
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_variant="epoch_first",
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:K1_STREAMSHIELD",
        "ECG_K1_STREAMSHIELD",
        "ECG:K1_SS",
        "ECG_K1_SS",
    ):
        return PolicySpec(
            label="ECG_K1_STREAMSHIELD",
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_stream_bypass=True,
            ecg_variant="epoch_first",
            ecg_transport_pinned=True,
        )
    if upper.startswith("ECG:"):
        mode = upper.split(":", 1)[1]
        label = f"ECG_{mode}" + ("_CHARGED" if charge_popt else "")
        return PolicySpec(
            label=label,
            policy="ECG",
            ecg_mode=mode,
            charge_popt_overhead=charge_popt,
        )
    if upper.startswith("ECG_") and upper != "ECG":
        mode = upper.split("ECG_", 1)[1]
        label = f"ECG_{mode}" + ("_CHARGED" if charge_popt else "")
        return PolicySpec(
            label=label,
            policy="ECG",
            ecg_mode=mode,
            charge_popt_overhead=charge_popt,
        )
    if upper in ("P_OPT", "POPT"):
        if not explicit_charge:
            charge_popt = True
        return PolicySpec(
            label="POPT" if charge_popt else "POPT_UNCHARGED",
            policy="POPT",
            charge_popt_overhead=charge_popt,
        )
    return PolicySpec(
        label=upper,
        policy=upper,
        charge_popt_overhead=charge_popt,
    )


def policy_output_label(text: str) -> str:
    return parse_policy_spec(text).label
