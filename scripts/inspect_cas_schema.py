"""Inspect CAS rxndata XML keys across many reactions to find catalyst fields."""
from pymongo import MongoClient
from collections import Counter

uri = "mongodb://mituser:trDwYgnFNCZSc22q@molgpu04.mit.edu:27018/?authSource=cas"
client = MongoClient(uri)

try:
    cas = client.get_database('cas')
    rxndata = cas['rxndata']

    # Sample many reactions and collect all keys at each level
    stage_keys = Counter()
    reaction_keys = Counter()

    for doc in rxndata.find({"num_products": {"$gte": 2}}).limit(500):
        rxn = doc["xmldata"]["reaction"]
        for k in rxn.keys():
            reaction_keys[k] += 1

        stages = rxn.get("reaction-stage", [])
        if isinstance(stages, dict):
            stages = [stages]
        for stage in stages:
            for k in stage.keys():
                stage_keys[k] += 1

    print("=== reaction-level keys (across 500 docs) ===")
    for k, v in reaction_keys.most_common():
        print(f"  {k}: {v}")

    print("\n=== reaction-stage keys (across all stages) ===")
    for k, v in stage_keys.most_common():
        print(f"  {k}: {v}")

finally:
    client.close()
