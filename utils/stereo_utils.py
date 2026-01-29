import numpy as np
from rdkit import Chem



try:
    from rdchiral.utils import parity4
except ImportError:
    # Fallback parity4 implementation if rdchiral not available
    def parity4(data):
        """
        Calculate parity of 4-element list (from rdchiral)
        Returns 0 for even permutation, 1 for odd permutation
        """
        if data[0] < data[1]:
            if data[2] < data[3]:
                if data[0] < data[2]:
                    if data[1] < data[2]:
                        return 0 # (0, 1, 2, 3) 
                    else:
                        if data[1] < data[3]:
                            return 1 # (0, 2, 1, 3) 
                        else:
                            return 0 # (0, 3, 1, 2) 
                else:
                    if data[0] < data[3]:
                        if data[1] < data[3]:
                            return 0 # (1, 2, 0, 3) 
                        else:
                            return 1 # (1, 3, 0, 2) 
                    else:
                        return 0 # (2, 3, 0, 1) 
            else:
                if data[0] < data[3]:
                    if data[1] < data[2]:
                        if data[1] < data[3]:
                            return 1 # (0, 1, 3, 2) 
                        else:
                            return 0 # (0, 2, 3, 1) 
                    else:
                        return 1 # (0, 3, 2, 1) 
                else:
                    if data[0] < data[2]:
                        if data[1] < data[2]:
                            return 1 # (1, 2, 3, 0) 
                        else:
                            return 0 # (1, 3, 2, 0) 
                    else:
                        return 1 # (2, 3, 1, 0) 
        else:
            if data[2] < data[3]:
                if data[0] < data[3]:
                    if data[0] < data[2]:
                        return 1 # (1, 0, 2, 3) 
                    else:
                        if data[1] < data[2]:
                            return 0 # (2, 0, 1, 3) 
                        else:
                            return 1 # (2, 1, 0, 3) 
                else:
                    if data[1] < data[2]:
                        return 1 # (3, 0, 1, 2) 
                    else:
                        if data[1] < data[3]:
                            return 0 # (3, 1, 0, 2) 
                        else:
                            return 1 # (3, 2, 0, 1) 
            else:
                if data[0] < data[2]:
                    if data[0] < data[3]:
                        return 0 # (1, 0, 3, 2) 
                    else:
                        if data[1] < data[3]:
                            return 1 # (2, 0, 3, 1) 
                        else:
                            return 0 # (2, 1, 3, 0) 
                else:
                    if data[1] < data[2]:
                        if data[1] < data[3]:
                            return 0 # (3, 0, 2, 1) 
                        else:
                            return 1 # (3, 1, 2, 0) 
                    else:
                        return 0 # (3, 2, 1, 0) 

def get_neighbor_map_nums(atom):
    """
    Get neighbor atom map numbers for parity calculation (from rdchiral)
    """
    map_nums = [bond.GetOtherAtom(atom).GetIntProp('molAtomMapNumber') 
                for bond in atom.GetBonds()]
    
    # Add hydrogen if degree < 4 (from rdchiral logic)
    if len(map_nums) < 4:
        map_nums.append(-1)  # H
    
    return map_nums

def atom_chirality_matches_rdchiral(a_reactant, a_product):
    """
    Check chirality consistency between reactant and product atoms using rdchiral's approach
    Returns: 1 if match, -1 if opposite, 0 if no match, 2 if ambiguous
    """
    if a_product.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        if a_reactant.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return 2  # achiral -> achiral
        return 0  # product achiral, reactant chiral -> no match
    
    if a_reactant.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        return 0  # reactant achiral, product chiral -> no match
    
    # Both have specified chirality - use rdchiral's parity approach
    mapnums_reactant = get_neighbor_map_nums(a_reactant)
    mapnums_product = get_neighbor_map_nums(a_product)
    
    # When there are fewer than 3 heavy neighbors, chirality is ambiguous
    if len(mapnums_reactant) < 3 or len(mapnums_product) < 3:
        return 2
    
    try:
        # Handle missing neighbors (rdchiral approach)
        only_in_reactant = [i for i in mapnums_reactant if i not in mapnums_product][::-1]
        only_in_product = [i for i in mapnums_product if i not in mapnums_reactant]
        
        if len(only_in_reactant) <= 1 and len(only_in_product) <= 1:
            reactant_parity = parity4(mapnums_reactant)
            product_parity = parity4([i if i in mapnums_reactant else only_in_reactant.pop() for i in mapnums_product])
            
            parity_matches = reactant_parity == product_parity
            tag_matches = a_reactant.GetChiralTag() == a_product.GetChiralTag()
            chirality_matches = parity_matches == tag_matches
            
            return 1 if chirality_matches else -1
        else:
            return 2  # ambiguous case
    except (IndexError, KeyError):
        return 2  # ambiguous case

def analyze_stereocenter_changes_parity(reactant_smiles, product_smiles):
    """
    Analyze stereocenter changes using rdchiral's proven parity approach
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True
    
    reactant_mol = Chem.MolFromSmiles(reactant_smiles, ps)
    product_mol = Chem.MolFromSmiles(product_smiles, ps)
    
    # Ensure stereochemistry is assigned
    Chem.AssignStereochemistry(reactant_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    Chem.AssignStereochemistry(product_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    
    changes = {}
    
    # Get all mapped atoms
    reactant_atoms = {atom.GetIntProp('molAtomMapNumber'): atom 
                     for atom in reactant_mol.GetAtoms() 
                     if atom.GetIntProp('molAtomMapNumber') > 0}
    
    product_atoms = {atom.GetIntProp('molAtomMapNumber'): atom 
                    for atom in product_mol.GetAtoms() 
                    if atom.GetIntProp('molAtomMapNumber') > 0}
    
    all_map_nums = set(reactant_atoms.keys()) | set(product_atoms.keys())
    
    for map_num in all_map_nums:
        reactant_atom = reactant_atoms.get(map_num)
        product_atom = product_atoms.get(map_num)
        
        if reactant_atom is None:
            changes[map_num] = 'created'
        elif product_atom is None:
            changes[map_num] = 'destroyed'
        else:
            # Both exist - check stereochemistry using rdchiral's approach
            reactant_chiral = reactant_atom.GetChiralTag()
            product_chiral = product_atom.GetChiralTag()
            
            if reactant_chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED and product_chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                changes[map_num] = 'no_stereo'
            elif reactant_chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED and product_chiral != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                changes[map_num] = 'stereo_created'
            elif reactant_chiral != Chem.rdchem.ChiralType.CHI_UNSPECIFIED and product_chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                changes[map_num] = 'stereo_destroyed'
            else:
                # Both have specified chirality - use rdchiral's matching logic
                match_result = atom_chirality_matches_rdchiral(reactant_atom, product_atom)
                
                if match_result == 1:
                    changes[map_num] = 'preserved'
                elif match_result == -1:
                    changes[map_num] = 'inverted'
                elif match_result == 0:
                    changes[map_num] = 'stereo_destroyed'  # No match
                else:  # match_result == 2 (ambiguous)
                    # Fall back to simple chiral tag comparison
                    if reactant_chiral != product_chiral:
                        changes[map_num] = 'inverted'
                    else:
                        changes[map_num] = 'preserved'
    
    return changes

def get_chiral_vector_parity(reactant_smiles, product_smiles):
    """
    Create enhanced chirality vector using rdchiral's parity approach
    """
    changes = analyze_stereocenter_changes_parity(reactant_smiles, product_smiles)
    
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True
    
    product_mol = Chem.MolFromSmiles(product_smiles, ps)
    Chem.AssignStereochemistry(product_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    
    n = len(product_mol.GetAtoms())
    v = np.zeros((n, 3), dtype=np.float16)
    
    for atom in product_mol.GetAtoms():
        map_num = atom.GetIntProp('molAtomMapNumber')
        idx = map_num - 1
        
        # Skip if index is out of bounds (shouldn't happen with proper atom mapping)
        if idx >= n or idx < 0:
            continue
        
        # Chirality value
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            v[idx, 0] = 1.0
        elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            v[idx, 0] = -1.0
        else:
            v[idx, 0] = 0.0
        
        # Change type
        change_type = changes.get(map_num, 'no_stereo')
        change_encoding = {
            'no_stereo': 0.0,
            'preserved': 1.0,
            'inverted': 2.0,
            'stereo_created': 3.0,
            'stereo_destroyed': 4.0,
            'created': 5.0,
            'destroyed': 6.0
        }
        v[idx, 1] = change_encoding.get(change_type, 0.0)
        
        # Confidence
        if change_type in ['preserved', 'inverted']:
            v[idx, 2] = 1.0  # High confidence
        elif change_type in ['stereo_created', 'stereo_destroyed']:
            v[idx, 2] = 0.8  # Medium confidence
        else:
            v[idx, 2] = 0.0  # Low confidence
    
    return v

def get_single_molecule_chiral_vector(smiles):
    """
    Create chirality vector for a single molecule (self-reference)
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True
    
    mol = Chem.MolFromSmiles(smiles, ps)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    
    n = len(mol.GetAtoms())
    v = np.zeros((n, 3), dtype=np.float16)
    
    for atom in mol.GetAtoms():
        map_num = atom.GetIntProp('molAtomMapNumber')
        idx = map_num - 1
        
        # Skip if index is out of bounds
        if idx >= n or idx < 0:
            continue
            
        chiral_tag = atom.GetChiralTag()
        
        if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            v[idx, 0] = 1.0
        elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            v[idx, 0] = -1.0
        else:
            v[idx, 0] = 0.0
        
        v[idx, 1] = 1.0  # preserved
        v[idx, 2] = 1.0  # high confidence
    
    return v