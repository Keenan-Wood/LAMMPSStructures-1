class BondStyle:
    def __init__(
        self,
        bond_type: str,
        bond_style: str,
        N_atoms: int,
        params: list
        ):

        self.type = bond_type
        self.style = bond_style
        self.N_atoms = N_atoms
        self.params = params

bond_style_list = []
bond_style_list.append(BondStyle(
    bond_type = 'bond',
    bond_style = 'harmonic',
    N_atoms = 2, 
    params = [('stiffness', float | int), ('rest_length', float | int)]
    ))
bond_style_list.append(BondStyle(
    bond_type = 'angle',
    bond_style = 'cosine/delta',
    N_atoms = 3,
    params = [('energy', float | int), ('offset', float | int)]
    ))
bond_style_list.append(BondStyle(
    bond_type = 'dihedral',
    bond_style = 'spherical',
    N_atoms = 4,
    params = [
        ('N_sub_params', int),
        ('sub_params', [('energy', float | int), ('frequency_phi', int), ('offset_phi', float | int), ('u', int),
                                           ('frequency_1', int), ('offset_1', float | int), ('v', int),
                                           ('frequency_2', int), ('offset_2', float | int), ('w', int)])
        ]
    ))