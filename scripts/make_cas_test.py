from pymongo import MongoClient
from rdkit import Chem
from collections import Counter
import sys
import os


def ensure_list(val):
    """XML-to-dict can produce a single dict or a list; normalize to list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def extract_ids(stage_entry, key):
    """Extract registry-number ints from a stage's reactant/reagent/solvent entry."""
    items = ensure_list(stage_entry.get(key))
    ids = []
    for item in items:
        uri = item.get("substance-uri", {})
        for u in ensure_list(uri):
            ids.append(int(u["@registry-number"]))
    return ids


def extract_product_ids(reaction):
    """Extract product registry-number ints from reaction-product."""
    products = ensure_list(reaction.get("reaction-product"))
    ids = []
    for prod in products:
        uri = prod.get("substance-uri", {})
        for u in ensure_list(uri):
            ids.append(int(u["@registry-number"]))
    return ids


def convert_dative_bonds(mol):
    """Convert dative bonds (<- / ->) to single bonds for BE matrix compatibility.
    Both are 2-electron bonds; FlowER only handles SINGLE/DOUBLE/TRIPLE/AROMATIC.
    Returns None if conversion fails."""
    has_dative = any(b.GetBondType() == Chem.rdchem.BondType.DATIVE
                     for b in mol.GetBonds())
    if not has_dative:
        return mol
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DATIVE:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except:
        try:
            # Metal complexes may fail valence checks — skip that part only
            Chem.SanitizeMol(mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except:
            return None
    return mol


_smi_params = Chem.SmilesWriteParams()
_smi_params.allHsExplicit = True
_smi_params.isomericSmiles = False
_smi_params.includeDativeBonds = False


def mol_to_explicit_h_smi(mol):
    """Add explicit Hs and return SMILES with all Hs shown.
    Returns (None, None) if conversion fails."""
    # Reject wildcard atoms — FlowER can't handle them
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None, None
    mol = Chem.AddHs(mol, explicitOnly=False)
    smi = Chem.MolToSmiles(mol, _smi_params)
    return smi, mol


def to_clean_smi(explicit_h_smi):
    """Convert explicit-H SMILES to clean canonical form (no Hs, no maps)."""
    mol = Chem.MolFromSmiles(explicit_h_smi)
    if mol is None:
        return explicit_h_smi
    mol = Chem.RemoveHs(mol)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def count_atoms(mol):
    """Count atoms by element (mol must already have explicit Hs)."""
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return counts


def try_balance(lhs_atom_counts, rhs_total, max_copies=5):
    """
    Find multipliers n_i (each 1..max_copies) for LHS molecules such that
    sum(n_i * lhs_atom_counts[i]) == rhs_total.

    Returns list of multipliers, or None if impossible.
    """
    k = len(lhs_atom_counts)
    multipliers = [1] * k

    # Current LHS total
    current = Counter()
    for c in lhs_atom_counts:
        current += c

    # deficit[elem] = how many more atoms the RHS has
    all_elems = set(current) | set(rhs_total)
    deficit = {e: rhs_total.get(e, 0) - current.get(e, 0) for e in all_elems}

    # If LHS already exceeds RHS for any element, can't fix by adding more LHS
    if any(v < 0 for v in deficit.values()):
        return None

    # Greedy: add copies of whichever molecule best reduces the deficit
    for _ in range(max_copies * k):
        if all(v == 0 for v in deficit.values()):
            return multipliers

        best_idx = None
        best_score = None
        best_new_deficit = None

        for i in range(k):
            if multipliers[i] >= max_copies:
                continue
            new_deficit = {e: deficit[e] - lhs_atom_counts[i].get(e, 0) for e in deficit}
            if any(v < 0 for v in new_deficit.values()):
                continue
            score = sum(new_deficit.values())
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i
                best_new_deficit = new_deficit

        if best_idx is None:
            return None
        multipliers[best_idx] += 1
        deficit = best_new_deficit

    if all(v == 0 for v in deficit.values()):
        return multipliers
    return None


uri = "mongodb://mituser:trDwYgnFNCZSc22q@molgpu04.mit.edu:27018/?authSource=cas"
client = MongoClient(uri)

try:
    cas = client.get_database('cas')
    rxndata = cas['rxndata']
    molecules = cas['molecules']

    # Fetch all reactions with >= 2 products
    reactions = list(rxndata.find(
        {"num_products": {"$gte": 2}},
        {"xmldata.reaction": 1, "_id": 1},
    ))
    print(f"Found {len(reactions)} reactions with >= 2 products", file=sys.stderr)

    # First pass: parse all reactions, collect molecule IDs
    parsed = []
    all_ids = set()
    for doc in reactions:
        rxn = doc["xmldata"]["reaction"]
        product_ids = extract_product_ids(rxn)

        reactant_ids = []
        solvent_ids = []
        reagent_ids = []
        catalyst_ids = []
        for stage in ensure_list(rxn.get("reaction-stage")):
            reactant_ids.extend(extract_ids(stage, "reaction-reactant"))
            solvent_ids.extend(extract_ids(stage, "reaction-solvent"))
            reagent_ids.extend(extract_ids(stage, "reaction-reagent"))
            catalyst_ids.extend(extract_ids(stage, "reaction-catalyst"))

        all_mol_ids = product_ids + reactant_ids + solvent_ids + reagent_ids + catalyst_ids
        all_ids.update(all_mol_ids)
        parsed.append((reactant_ids, solvent_ids, reagent_ids, catalyst_ids, product_ids))

    print(f"Looking up {len(all_ids)} unique molecules", file=sys.stderr)

    # Single bulk lookup — store explicit-H SMILES and atom counts
    id_to_smi = {}
    id_to_atoms = {}
    all_ids_list = list(all_ids)
    CHUNK = 50000
    for i in range(0, len(all_ids_list), CHUNK):
        chunk = all_ids_list[i:i + CHUNK]
        for doc in molecules.find({"_id": {"$in": chunk}}, {"smiles": 1}):
            mol = Chem.MolFromSmiles(doc["smiles"])
            if mol is not None:
                smi, mol_h = mol_to_explicit_h_smi(mol)
                if smi is not None:
                    id_to_smi[doc["_id"]] = smi
                    id_to_atoms[doc["_id"]] = count_atoms(mol_h)

    print(f"Resolved {len(id_to_smi)}/{len(all_ids)} molecules", file=sys.stderr)

    # Second pass: balance each product INDIVIDUALLY against reactants/reagents.
    # Each product is a separate reaction from the same starting materials.
    output_lines = []
    target_lines = []  # clean canonical target products (for post-hoc checking)
    n_complete = 0
    n_incomplete = 0
    n_prod_balanced = 0
    n_prod_unbalanced = 0

    for reactant_ids, solvent_ids, reagent_ids, catalyst_ids, product_ids in parsed:
        # Reactants/reagents must all resolve
        core_ids = reactant_ids + reagent_ids
        if not all(mid in id_to_smi for mid in core_ids):
            n_incomplete += 1
            continue
        if not core_ids:
            n_incomplete += 1
            continue

        resolved_solvent_ids = [mid for mid in solvent_ids if mid in id_to_smi]
        resolved_catalyst_ids = [mid for mid in catalyst_ids if mid in id_to_smi]
        spectator_smiles = ([id_to_smi[mid] for mid in resolved_solvent_ids]
                            + [id_to_smi[mid] for mid in resolved_catalyst_ids])

        # Balance each product individually against ONLY core (reactants+reagents).
        # Solvents/catalysts are spectators — not part of balancing.
        core_atom_counts = [id_to_atoms[mid] for mid in core_ids]
        max_mults = [1] * len(core_ids)

        good_products = []
        seen_smiles = set()
        for pid in product_ids:
            if pid not in id_to_smi:
                continue
            psmi = id_to_smi[pid]
            if psmi in seen_smiles:
                continue
            seen_smiles.add(psmi)

            rhs_atoms = id_to_atoms[pid]
            mults = try_balance(core_atom_counts, rhs_atoms, max_copies=5)
            if mults is not None:
                for i, m in enumerate(mults):
                    max_mults[i] = max(max_mults[i], m)
                n_prod_balanced += 1
                good_products.append(pid)
            else:
                n_prod_unbalanced += 1

        if len(good_products) < 2:
            n_incomplete += 1
            continue

        # LHS: max-multiplier copies of core + spectators (solvent, catalyst)
        lhs_smiles = []
        for mid, n in zip(core_ids, max_mults):
            lhs_smiles.extend([id_to_smi[mid]] * n)
        lhs_smiles.extend(spectator_smiles)

        # RHS: each outcome is product.solvent.catalyst, outcomes pipe-separated
        rhs_entries = []
        for pid in good_products:
            outcome = [id_to_smi[pid]] + spectator_smiles
            rhs_entries.append(".".join(outcome))

        output_lines.append(f"{'.'.join(lhs_smiles)}>>{'|'.join(rhs_entries)}")
        target_lines.append("|".join(to_clean_smi(id_to_smi[pid]) for pid in good_products))
        n_complete += 1

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cas")
    os.makedirs(out_dir, exist_ok=True)

    beam_path = os.path.join(out_dir, "beam.txt")
    with open(beam_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    targets_path = os.path.join(out_dir, "beam_targets.txt")
    with open(targets_path, "w") as f:
        for line in target_lines:
            f.write(line + "\n")

    print(f"\nWrote {n_complete} reactions to {beam_path}", file=sys.stderr)
    print(f"Wrote target products to {targets_path}", file=sys.stderr)
    print(f"Products: {n_prod_balanced} balanced, {n_prod_unbalanced} unbalanced (excluded)", file=sys.stderr)
    print(f"Skipped {n_incomplete} incomplete (missing molecules or <2 resolved products)", file=sys.stderr)

finally:
    client.close()
