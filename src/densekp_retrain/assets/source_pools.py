"""Metadata-only checks for proposed source pools. No asset or renderer imports."""

from __future__ import annotations

from typing import Any

SCHEMA = "densekp_source_pools_v1"
EPISODE_SCHEMA = "densekp_source_episode_candidate_v1"


def _issue(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def validate_proposal(catalog: dict, proposals: dict, proposal_id: str) -> list[dict]:
    issues: list[dict] = []
    if catalog.get("schema") != SCHEMA or proposals.get("schema") != SCHEMA:
        return [_issue("wrong_version", "catalog/proposal schema must be v1")]
    proposal = proposals.get("proposals", {}).get(proposal_id)
    if proposal is None:
        return [_issue("unknown_proposal", proposal_id)]
    combos = catalog.get("combinations", {})
    motions = catalog.get("motions", {})
    bodies = catalog.get("bodies", {})
    ownership: dict[str, str] = {}
    for side in ("train_candidate", "val_candidate"):
        for cid in proposal.get(side, []):
            c = combos.get(cid)
            if c is None:
                issues.append(_issue("dangling_combo", cid))
                continue
            b, m = bodies.get(c.get("body_id")), motions.get(c.get("motion_id"))
            if b is None or m is None:
                issues.append(_issue("dangling_component", cid))
                continue
            for key in ("character_family", "appearance_conservative_group"):
                val = b.get(key)
                if val:
                    group = f"{key}:{val}"
                    if group in ownership and ownership[group] != side:
                        issues.append(_issue("cross_pool_identity", group))
                    ownership[group] = side
            for key in ("recording_group", "participant_group"):
                val = m.get(key)
                if val and val != "unknown":
                    group = f"{key}:{val}"
                    if group in ownership and ownership[group] != side:
                        issues.append(_issue("cross_pool_motion", group))
                    ownership[group] = side
            if m.get("license_status") == "unknown" and proposal.get("formal_training_candidate"):
                issues.append(_issue("license_unresolved", cid))
    if proposal.get("claims_actor_independent_walking") and catalog.get("unknowns", {}).get("KIT11_vs_KIT425_actor") == "unknown":
        issues.append(_issue("actor_independence_unknown", "KIT11 versus KIT425"))
    return _unique(issues)


def _unique(issues: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for issue in issues:
        key = (issue["code"], issue["detail"])
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return out


def validate_episode(catalog: dict, proposals: dict, episode: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict] = []
    if episode.get("schema") != EPISODE_SCHEMA:
        issues.append(_issue("wrong_version", "episode schema must be v1"))
    pid, side = episode.get("proposal_id"), episode.get("pool")
    proposal = proposals.get("proposals", {}).get(pid)
    if not proposal or side not in ("train_candidate", "val_candidate"):
        issues.append(_issue("unknown_pool", f"{pid}/{side}"))
    if not episode.get("scene_episode_id") or not episode.get("rig_id"):
        issues.append(_issue("missing_episode_key", "scene_episode_id and rig_id required"))
    persons = episode.get("persons")
    if not isinstance(persons, list) or not persons:
        issues.append(_issue("missing_persons", "every target and occluder required"))
        persons = []
    if episode.get("expected_person_count") != len(persons):
        issues.append(_issue("person_count_mismatch", "declare every target and occluder"))
    results = []
    for person in persons:
        cid = person.get("combo_id")
        combo = catalog.get("combinations", {}).get(cid)
        pi: list[dict] = []
        if combo is None:
            pi.append(_issue("dangling_combo", str(cid)))
            results.append({"combo_id": cid, "issues": pi})
            issues.extend(pi)
            continue
        if person.get("role") not in ("target", "occluder"):
            pi.append(_issue("missing_role", str(cid)))
        for field in ("body_id", "motion_id", "clothing_id", "hair_id"):
            if person.get(field) != combo.get(field):
                pi.append(_issue("component_mismatch", f"{cid}/{field}"))
        for field, kind in (("body_id", "bodies"), ("motion_id", "motions"), ("clothing_id", "clothing"), ("hair_id", "hair")):
            if combo.get(field) not in catalog.get(kind, {}):
                pi.append(_issue("dangling_component", f"{cid}/{field}"))
        motion = catalog.get("motions", {}).get(combo.get("motion_id"), {})
        if person.get("declared_source_group") != motion.get("source_group"):
            pi.append(_issue("source_mismatch", str(cid)))
        window = person.get("asset_window")
        full = combo.get("asset_window")
        if not (isinstance(window, list) and len(window) == 2 and all(isinstance(x, int) for x in window)
                and isinstance(full, list) and len(full) == 2 and full[0] <= window[0] < window[1] <= full[1]):
            pi.append(_issue("invalid_window", str(cid)))
        if combo.get("fixed_cache_lock"):
            lock = combo["fixed_cache_lock"]
            if any(person.get(k) != lock.get(k) for k in ("body_id", "motion_id", "hair_id")):
                pi.append(_issue("fixed_cache_lock", str(cid)))
            if full != lock.get("authority_window"):
                pi.append(_issue("fixed_cache_authority", str(cid)))
        if proposal and cid not in proposal.get(side, []):
            pi.append(_issue("outside_pool", f"{cid} in {side}"))
        if proposal and proposal.get("formal_training_candidate") and motion.get("license_status") == "unknown":
            pi.append(_issue("license_unresolved", str(cid)))
        accepted = combo.get("human_accepted_scope")
        scope_ok = bool(accepted and isinstance(window, list) and accepted[0] <= window[0] and window[1] <= accepted[1])
        results.append({
            "combo_id": cid,
            "compatible": bool(combo.get("compatible")),
            "built": bool(combo.get("built")),
            "human_accepted_scope": scope_ok,
            "scene_consumed": bool(combo.get("scene_consumed")),
            "eligible_under_proposed_policy": not pi and bool(combo.get("compatible")) and bool(combo.get("built")) and scope_ok,
            "issues": pi,
        })
        issues.extend(pi)
    issues.extend(validate_proposal(catalog, proposals, pid) if proposal else [])
    return {
        "schema": "densekp_source_pool_validation_v1",
        "proposal_id": pid,
        "pool": side,
        "scene_episode_id": episode.get("scene_episode_id"),
        "rig_id": episode.get("rig_id"),
        "status": "candidate_only_pending_review" if not issues else "rejected_or_unresolved",
        "eligible_under_proposed_policy": not issues and all(r.get("eligible_under_proposed_policy") for r in results),
        "formal_training_approved": False,
        "ready_for_sampling": False,
        "persons": results,
        "issues": _unique(issues),
    }


def validate_batch(catalog: dict, proposals: dict, episodes: list[dict]) -> list[dict]:
    results = [validate_episode(catalog, proposals, e) for e in episodes]
    keys: dict[str, tuple[str, str]] = {}
    for result in results:
        episode_id = result.get("scene_episode_id")
        destination = (result.get("proposal_id"), result.get("pool"))
        if episode_id in keys and keys[episode_id] != destination:
            result["issues"].append(_issue("episode_cross_pool", str(episode_id)))
            result["eligible_under_proposed_policy"] = False
            result["status"] = "rejected_or_unresolved"
        else:
            keys[episode_id] = destination
    return results
