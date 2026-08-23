"""Canonical ECG experiment policy parsing and output labels."""

from __future__ import annotations

import re
from dataclasses import dataclass


ONLINE_DUELING_WINDOW_MISSES = 1024
REUSEPLAN_LABEL = "ECG_REUSEPLAN"
REUSEPLAN_GRASP_LABEL = "ECG_REUSEPLAN_GRASP"
REUSEPLAN_EPOCH_LABEL = "ECG_REUSEPLAN_EPOCH"
REUSEPLAN_RRIP_LABEL = "ECG_REUSEPLAN_RRIP"
REUSEPLAN_DEGREE_LABEL = "ECG_REUSEPLAN_DEGREE"
REUSEPLAN_LRU_LABEL = "ECG_REUSEPLAN_LRU"
REUSEPLAN_ONLINE_LABEL = "ECG_REUSEPLAN_ONLINE"
REUSEPLAN_FLOWTHROUGH_LABEL = "ECG_REUSEPLAN_FLOWTHROUGH"
REUSEPLAN_RRIP_FLOWTHROUGH_LABEL = "ECG_REUSEPLAN_RRIP_FLOWTHROUGH"
REUSEPLAN_LRU_FLOWTHROUGH_LABEL = "ECG_REUSEPLAN_LRU_FLOWTHROUGH"
REUSEPLAN_ONLINE_FLOWTHROUGH_LABEL = "ECG_REUSEPLAN_ONLINE_FLOWTHROUGH"
REUSEPLAN_ADAPTIVE_FLOWTHROUGH_LABEL = (
    "ECG_REUSEPLAN_ADAPTIVE_FLOWTHROUGH")
REUSEPLAN_ONLINE_ADAPTIVE_FLOWTHROUGH_LABEL = (
    "ECG_REUSEPLAN_ONLINE_ADAPTIVE_FLOWTHROUGH")
REUSEPLAN_SINGLE_EPOCH_LABEL = "ECG_REUSEPLAN_SINGLE_EPOCH"
REUSEPLAN_SINGLE_EPOCH_FLOWTHROUGH_LABEL = (
    "ECG_REUSEPLAN_SINGLE_EPOCH_FLOWTHROUGH")
ONLINE_DUELING_REQUIRED_POSITIVE_FIELDS = (
    "gem5_k2_dueling_request_bound_victims",
    "gem5_k2_dueling_leader_samples",
    "gem5_k2_dueling_follower_selections",
    "gem5_k2_dueling_completed_windows",
)
ONLINE_DUELING_REPORTED_FIELDS = (
    *ONLINE_DUELING_REQUIRED_POSITIVE_FIELDS,
    "gem5_k2_dueling_winner_changes",
    "gem5_k2_dueling_follower_variant_overrides",
)

# Sniper analog of the gem5 online-dueling evidence above. Field names use a
# "sniper_" prefix and "governed_victims" rather than gem5's frozen
# "request_bound_victims": Sniper has no O3 Request/MSHR-attested victim to
# bind to, so its population is the closest Sniper-equivalent (a
# marker/sideband-governed miss population; see cache_set_ecg.cc's
# OnlineDuelingEvidence comment and sniper_k2_dueling_binding_model). The
# frozen gem5_* fields above are never renamed or repurposed for Sniper.
SNIPER_ONLINE_DUELING_REQUIRED_POSITIVE_FIELDS = (
    "sniper_k2_dueling_governed_victims",
    "sniper_k2_dueling_leader_samples",
    "sniper_k2_dueling_follower_selections",
    "sniper_k2_dueling_completed_windows",
)
SNIPER_ONLINE_DUELING_REPORTED_FIELDS = (
    *SNIPER_ONLINE_DUELING_REQUIRED_POSITIVE_FIELDS,
    "sniper_k2_dueling_winner_changes",
    "sniper_k2_dueling_follower_variant_overrides",
)


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy: str
    ecg_mode: str | None = None
    charge_popt_overhead: bool = False
    ecg_schedule_k: int = 0
    ecg_stream_bypass: bool = False
    ecg_stream_adaptive: bool = False
    ecg_variant: str | None = None
    ecg_transport_pinned: bool = False
    ecg_set_dueling: bool = False

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

    if upper in (
        "ECG:REUSEPLAN", "ECG_REUSEPLAN",
        "ECG:K2", "ECG_K2",
    ):
        return PolicySpec(
            label=REUSEPLAN_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_variant="adaptive",
            ecg_transport_pinned=True,
        )
    k2_variants = {
        "ECG:REUSEPLAN_GRASP": (REUSEPLAN_GRASP_LABEL, "grasp_only"),
        "ECG_REUSEPLAN_GRASP": (REUSEPLAN_GRASP_LABEL, "grasp_only"),
        "ECG:K2_GRASP": (REUSEPLAN_GRASP_LABEL, "grasp_only"),
        "ECG_K2_GRASP": (REUSEPLAN_GRASP_LABEL, "grasp_only"),
        "ECG:REUSEPLAN_EPOCH": (REUSEPLAN_EPOCH_LABEL, "epoch_first"),
        "ECG_REUSEPLAN_EPOCH": (REUSEPLAN_EPOCH_LABEL, "epoch_first"),
        "ECG:K2_EPOCH": (REUSEPLAN_EPOCH_LABEL, "epoch_first"),
        "ECG_K2_EPOCH": (REUSEPLAN_EPOCH_LABEL, "epoch_first"),
        "ECG:REUSEPLAN_RRIP": (REUSEPLAN_RRIP_LABEL, "rrip_first"),
        "ECG_REUSEPLAN_RRIP": (REUSEPLAN_RRIP_LABEL, "rrip_first"),
        "ECG:K2_RRIP": (REUSEPLAN_RRIP_LABEL, "rrip_first"),
        "ECG_K2_RRIP": (REUSEPLAN_RRIP_LABEL, "rrip_first"),
        "ECG:REUSEPLAN_DEGREE": (REUSEPLAN_DEGREE_LABEL, "degree_first"),
        "ECG_REUSEPLAN_DEGREE": (REUSEPLAN_DEGREE_LABEL, "degree_first"),
        "ECG:K2_DEGREE": (REUSEPLAN_DEGREE_LABEL, "degree_first"),
        "ECG_K2_DEGREE": (REUSEPLAN_DEGREE_LABEL, "degree_first"),
        "ECG:REUSEPLAN_LRU": (REUSEPLAN_LRU_LABEL, "lru_only"),
        "ECG_REUSEPLAN_LRU": (REUSEPLAN_LRU_LABEL, "lru_only"),
        "ECG:K2_LRU": (REUSEPLAN_LRU_LABEL, "lru_only"),
        "ECG_K2_LRU": (REUSEPLAN_LRU_LABEL, "lru_only"),
    }
    if upper in k2_variants:
        label, variant = k2_variants[upper]
        return PolicySpec(
            label=label,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_variant=variant,
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:REUSEPLAN_RRIP_FLOWTHROUGH",
        "ECG_REUSEPLAN_RRIP_FLOWTHROUGH",
        "ECG:REUSEPLAN_RRIP_FT",
        "ECG_REUSEPLAN_RRIP_FT",
        "ECG:K2_RRIP_STREAMSHIELD",
        "ECG_K2_RRIP_STREAMSHIELD",
        "ECG:K2_RRIP_SS",
        "ECG_K2_RRIP_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_RRIP_FLOWTHROUGH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_variant="rrip_first",
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:REUSEPLAN_LRU_FLOWTHROUGH",
        "ECG_REUSEPLAN_LRU_FLOWTHROUGH",
        "ECG:REUSEPLAN_LRU_FT",
        "ECG_REUSEPLAN_LRU_FT",
        "ECG:K2_LRU_STREAMSHIELD",
        "ECG_K2_LRU_STREAMSHIELD",
        "ECG:K2_LRU_SS",
        "ECG_K2_LRU_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_LRU_FLOWTHROUGH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_variant="lru_only",
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:REUSEPLAN_FLOWTHROUGH",
        "ECG_REUSEPLAN_FLOWTHROUGH",
        "ECG:REUSEPLAN_FT",
        "ECG_REUSEPLAN_FT",
        "ECG:K2_STREAMSHIELD",
        "ECG_K2_STREAMSHIELD",
        "ECG:K2_SS",
        "ECG_K2_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_FLOWTHROUGH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_variant="adaptive",
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:REUSEPLAN_ONLINE", "ECG_REUSEPLAN_ONLINE",
        "ECG:K2_ONLINE", "ECG_K2_ONLINE",
    ):
        return PolicySpec(
            label=REUSEPLAN_ONLINE_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_variant="rrip_first",
            ecg_transport_pinned=True,
            ecg_set_dueling=True,
        )
    if upper in (
        "ECG:REUSEPLAN_ONLINE_FLOWTHROUGH",
        "ECG_REUSEPLAN_ONLINE_FLOWTHROUGH",
        "ECG:REUSEPLAN_ONLINE_FT",
        "ECG_REUSEPLAN_ONLINE_FT",
        "ECG:K2_ONLINE_STREAMSHIELD",
        "ECG_K2_ONLINE_STREAMSHIELD",
        "ECG:K2_ONLINE_SS",
        "ECG_K2_ONLINE_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_ONLINE_FLOWTHROUGH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_variant="rrip_first",
            ecg_transport_pinned=True,
            ecg_set_dueling=True,
        )
    if upper in (
        "ECG:REUSEPLAN_ADAPTIVE_FLOWTHROUGH",
        "ECG_REUSEPLAN_ADAPTIVE_FLOWTHROUGH",
        "ECG:REUSEPLAN_ADAPTIVE_FT",
        "ECG_REUSEPLAN_ADAPTIVE_FT",
        "ECG:K2_ADAPTIVE_STREAMSHIELD",
        "ECG_K2_ADAPTIVE_STREAMSHIELD",
        "ECG:K2_ADAPTIVE_SS",
        "ECG_K2_ADAPTIVE_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_ADAPTIVE_FLOWTHROUGH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_stream_adaptive=True,
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:REUSEPLAN_ONLINE_ADAPTIVE_FLOWTHROUGH",
        "ECG_REUSEPLAN_ONLINE_ADAPTIVE_FLOWTHROUGH",
        "ECG:REUSEPLAN_ONLINE_ADAPTIVE_FT",
        "ECG_REUSEPLAN_ONLINE_ADAPTIVE_FT",
        "ECG:K2_ONLINE_ADAPTIVE_STREAMSHIELD",
        "ECG_K2_ONLINE_ADAPTIVE_STREAMSHIELD",
        "ECG:K2_ONLINE_ADAPTIVE_SS",
        "ECG_K2_ONLINE_ADAPTIVE_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_ONLINE_ADAPTIVE_FLOWTHROUGH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_schedule_k=2,
            ecg_stream_bypass=True,
            ecg_stream_adaptive=True,
            ecg_variant="rrip_first",
            ecg_transport_pinned=True,
            ecg_set_dueling=True,
        )
    if upper in (
        "ECG:REUSEPLAN_SINGLE_EPOCH", "ECG_REUSEPLAN_SINGLE_EPOCH",
        "ECG:K1", "ECG_K1",
    ):
        return PolicySpec(
            label=REUSEPLAN_SINGLE_EPOCH_LABEL,
            policy="ECG",
            ecg_mode="ECG_GRASP_POPT",
            ecg_variant="epoch_first",
            ecg_transport_pinned=True,
        )
    if upper in (
        "ECG:REUSEPLAN_SINGLE_EPOCH_FLOWTHROUGH",
        "ECG_REUSEPLAN_SINGLE_EPOCH_FLOWTHROUGH",
        "ECG:REUSEPLAN_SINGLE_EPOCH_FT",
        "ECG_REUSEPLAN_SINGLE_EPOCH_FT",
        "ECG:K1_STREAMSHIELD",
        "ECG_K1_STREAMSHIELD",
        "ECG:K1_SS",
        "ECG_K1_SS",
    ):
        return PolicySpec(
            label=REUSEPLAN_SINGLE_EPOCH_FLOWTHROUGH_LABEL,
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
    if upper in ("HAWKEYE_PROXY", "HAWKEYE:PROXY"):
        return PolicySpec(
            label="HAWKEYE_PROXY",
            policy="HAWKEYE",
            charge_popt_overhead=False,
        )
    if upper == "HAWKEYE":
        return PolicySpec(
            label="HAWKEYE",
            policy="HAWKEYE",
            charge_popt_overhead=False,
        )
    return PolicySpec(
        label=upper,
        policy=upper,
        charge_popt_overhead=charge_popt,
    )


def policy_output_label(text: str) -> str:
    return parse_policy_spec(text).label


def is_reuseplan_policy(text: str) -> bool:
    return policy_output_label(text).startswith("ECG_REUSEPLAN")


def is_flowthrough_policy(text: str) -> bool:
    return policy_output_label(text).endswith("_FLOWTHROUGH")
