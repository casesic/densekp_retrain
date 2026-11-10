#!/usr/bin/env python3
"""Check a proposed source-pool episode using metadata only."""

import argparse
import json
from pathlib import Path

from densekp_retrain.assets.source_pools import validate_batch, validate_episode, validate_proposal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--proposal-id")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--example-id", help="key in a named example-episodes JSON object")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    pools = json.loads(args.pools.read_text())
    if args.episode:
        episode = json.loads(args.episode.read_text())
        if args.example_id:
            if not isinstance(episode, dict) or args.example_id not in episode:
                parser.error("unknown --example-id")
            episode = episode[args.example_id]
        if isinstance(episode, list):
            result = validate_batch(catalog, pools, episode)
            failed = any(x["issues"] for x in result)
        else:
            if not isinstance(episode, dict) or episode.get("schema") != "densekp_source_episode_candidate_v1":
                parser.error("--episode must be one episode, a list, or use --example-id for named examples")
            result = validate_episode(catalog, pools, episode)
            failed = bool(result["issues"])
    else:
        if not args.proposal_id:
            parser.error("--proposal-id is required without --episode")
        result = {"proposal_id": args.proposal_id, "issues": validate_proposal(catalog, pools, args.proposal_id)}
        failed = bool(result["issues"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
