"""
Check product coverage from beam search results on CAS multi-product reactions.

A product is "found" when a terminal node (no-op self-loop in the beam search
graph — meaning FlowER predicted no further reaction) contains that product
as a component.

Usage:
    python scripts/check_cas_coverage.py \
        --result_dir results/cas/ \
        --targets data/cas/beam_targets.txt

Works with outputs from both beam_predict_multiGPU.py (result_chunk_*.pickle)
and beam_predict_diverse_multiGPU.py (result_diverse_chunk_*.pickle).
"""

import os
import sys
import re
import pickle
import argparse
from rdkit import Chem
import networkx as nx


def clean(smi):
    """Canonicalize SMILES: remove Hs, clear atom maps."""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except:
        return None


def terminal_products(graph):
    """Canonical SMILES of molecule components in terminal nodes (self-loop = no-op)."""
    found = set()
    for node in nx.nodes_with_selfloops(graph):
        c = clean(node)
        if c:
            for part in c.split('.'):
                found.add(part)
    return found


def all_node_products(graph):
    """Canonical SMILES of molecule components across ALL nodes (for partial-credit)."""
    found = set()
    for node in graph.nodes():
        c = clean(node)
        if c:
            for part in c.split('.'):
                found.add(part)
    return found


def main():
    parser = argparse.ArgumentParser(description="Check CAS beam search product coverage")
    parser.add_argument("--result_dir", required=True,
                        help="Directory with result pickle files")
    parser.add_argument("--targets", required=True,
                        help="beam_targets.txt (one line per reaction, pipe-separated products)")
    args = parser.parse_args()

    # Load targets: each line is "prod1|prod2|..." in clean canonical form
    with open(args.targets) as f:
        targets = [set(line.strip().split("|")) for line in f if line.strip()]
    print(f"Loaded {len(targets)} reactions from {args.targets}")

    # Load pickle results in chunk order
    pfiles = sorted(
        [f for f in os.listdir(args.result_dir) if f.endswith('.pickle')],
        key=lambda f: int(re.search(r'(\d+)', f.split('chunk')[-1]).group(1))
            if re.search(r'(\d+)', f.split('chunk')[-1]) else 0
    )
    if not pfiles:
        print(f"ERROR: No pickle files found in {args.result_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for pf in pfiles:
        with open(os.path.join(args.result_dir, pf), 'rb') as f:
            results.extend(pickle.load(f))
    print(f"Loaded {len(results)} beam search results from {len(pfiles)} files")

    n = min(len(targets), len(results))
    if len(targets) != len(results):
        print(f"WARNING: {len(targets)} targets vs {len(results)} results")

    # Check coverage
    total_prods = 0
    total_terminal = 0
    total_any = 0
    details = []

    for i in range(n):
        graph, root, (reactant, _), _ = results[i]
        target_set = targets[i]

        term = terminal_products(graph)
        all_n = all_node_products(graph)

        hits_term = target_set & term
        hits_any = target_set & all_n

        total_prods += len(target_set)
        total_terminal += len(hits_term)
        total_any += len(hits_any)

        details.append({
            'idx': i,
            'reactant': reactant,
            'n_targets': len(target_set),
            'hits_terminal': len(hits_term),
            'hits_any': len(hits_any),
            'found_terminal': hits_term,
            'missed': target_set - term,
            'depth': graph.nodes[root].get('depth', 0),
            'n_nodes': graph.number_of_nodes(),
        })

    # Summary
    sep = '=' * 70
    print(f"\n{sep}")
    print(f"TERMINAL (no-op) coverage: {total_terminal}/{total_prods} products "
          f"({100 * total_terminal / max(total_prods, 1):.1f}%)")
    print(f"ANY-NODE coverage:         {total_any}/{total_prods} products "
          f"({100 * total_any / max(total_prods, 1):.1f}%)")

    full = sum(1 for d in details if d['hits_terminal'] == d['n_targets'])
    partial = sum(1 for d in details if 0 < d['hits_terminal'] < d['n_targets'])
    zero = sum(1 for d in details if d['hits_terminal'] == 0)

    print(f"\nReactions — full coverage:    {full}/{n} ({100 * full / max(n, 1):.1f}%)")
    print(f"Reactions — partial coverage: {partial}/{n} ({100 * partial / max(n, 1):.1f}%)")
    print(f"Reactions — zero coverage:    {zero}/{n} ({100 * zero / max(n, 1):.1f}%)")

    # Per-reaction details
    print(f"\n{sep}")
    print("Per-reaction breakdown:\n")
    for d in details:
        tag = ("FULL" if d['hits_terminal'] == d['n_targets']
               else "PART" if d['hits_terminal'] > 0 else "MISS")
        r = d['reactant'][:70] + "..." if len(d['reactant']) > 70 else d['reactant']
        print(f"  [{tag}] {d['idx']:4d}: {d['hits_terminal']}/{d['n_targets']} "
              f"terminal (any: {d['hits_any']})  depth={d['depth']}  "
              f"nodes={d['n_nodes']}  {r}")
        for m in sorted(d['missed']):
            print(f"           missed: {m}")


if __name__ == "__main__":
    main()
