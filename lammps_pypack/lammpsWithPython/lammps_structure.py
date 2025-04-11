import copy as cp
import numpy as np

## General TODO:
## bond, structure, atom - Refactor for more robust/clearer bond and atom type handling
## bond, structure - Create function to calculate and format bond parameters (use for add_node_bonds too)
## structure, node, atom - Refactor constraints in a more lammps-friendly format

class Structure:
    '''
    An object representing a 3D structure or truss.

    Attributes:
        nodes - list of node objects
        elements - list of element objects representing links between nodes
        materials - list of material objects storing material properties
        xsections - list of cross section objects storing cross section properties
        atoms - list of atom objects to use as lammps atoms
        atom_start_id - ID of first atom in structure

    Methods:
        __init__ - creates xsec and element objects as defined by input lists to initialize a structure instance
        __str__ - generates a string representing an instance's data for easy printing
        __add__ - 
        __radd__ -
        __iadd__ - 

        add_nodes - Create new nodes and add to structure

        add_constraints - Fix DOF of specified node DOF

        add_materials - Add new or existing materials to structure
        
        add_xsections - Add new or existing cross sections (xsection objects) to structure

        add_elements - Add new elements defined between nodes and add to structure

        discretize(atom_distance: float) - interpolates coordinates along elements based on provided atom_distance
          - returns data needed to add the structure geometry to a Simulation instance
    '''
    def __init__(
        self,
        node_list: list | None = None,
        *,
        constraint_list: list | None = None,
        material_list: list | None = None,
        xsection_list: list | None = None,
        element_list: list | None = None,
        element_connections: list | None = None
        ):
        '''
        Arguments:
         - node_list - A list of node coordinates (# Nodes x # coords) or a list of nodes
         - constraint_list - A list of node coordinates to fix
           * in format (Node ID, DOF_1, DOF_2...) where DOF_1 - DOF_6 are 1 for fixed and 0 for free (default)
         - material_list - A list of materials to assign or of material properties to create materials
         - xsection_list - A list of cross sections to assign or of cross section properties to create cross sections
         - element_list - A list of elements specified by their end nodes
         - element_connections - A list of bonds used to join atoms from pairs of elements through their connecting node
           * by default, only 'single/bond' type bonds are used to connect element end atoms to node atoms
           * entry format: [[node_0, node_1, node_3], bond_type: str, bond_parameters: list]
        '''

        # Create nodes and set fixed DOF
        self.nodes = []
        if not node_list is None: self.add_nodes(node_list)
        if not constraint_list is None: self.add_constraints(constraint_list)

        # Create materials, cross sections, and elements with provided data; Add to structure
        self.materials = []
        if not material_list is None: self.add_materials(material_list)
        self.xsections = []
        if not xsection_list is None: self.add_xsections(xsection_list)

        # Create elements with provided element data; Add to 'elements' list; add to connected nodes
        self.elements = []
        if not element_list is None: self.add_elements(element_list)

        for nd in self.nodes: nd.assign_connected_elements()

        self.atoms = []
        self.bonds = []
        self.element_connections = element_connections

    def __str__(self) -> str:
        print_str = f'\nStructure Data\n'
        print_str += f'\n\nNodes:'
        for nd in self.nodes: print_str += f'\n  {nd}'
        print_str += f'\nMaterials:'
        for mat in self.materials: print_str += f'\n  {mat}'
        print_str += f'\n\nX-Sections:'
        for xsec in self.xsections: print_str += f'\n  {xsec}'
        print_str += f'\n\nElements:'
        for el in self.elements: print_str += f'\n  {el}'
        print_str += f'\n\nNumber of Atoms in Structure: {len(self.atoms)}\n'
        return print_str

    def __add__(self, structure_2):
        if not isinstance(structure_2, Structure): raise Exception('Only structure instances can be combined')
        new_structure_nodes = []
        new_structure_materials = []
        new_structure_xsections = []
        new_structure_elements = []

        node_ids = []
        for nd in self.nodes + structure_2.nodes:
            # Add node to list of new structure's nodes if ID is not already taken
            if not nd.id in node_ids:
                new_structure_nodes.append(nd)
                node_ids.append(nd.id)
            else:
                # Verify that nodes to merge are identical
                dup_node = next(filter(lambda new_node: new_node.id == nd.id, new_structure_nodes), None)
                if not all(nd.coords == dup_node.coords): raise Exception('Coordinates of nodes to merge do not match')
                if not all(nd.fixed_dof == dup_node.fixed_dof): raise Exception('Constraints of nodes to merge do not match')

        material_names = []
        for mat in self.materials + structure_2.materials:
            # Add material to list of new structure's materials if its name is not already taken
            if not mat.name in material_names:
                new_structure_materials.append(mat)
                material_ids.append(mat.name)
            else:
                # Verify that materials to merge are identical
                dup_mat = next(filter(lambda new_mat: new_mat.name == mat.name, new_structure_materials), None)
                if not mat.E == dup_mat.E: raise Exception('Young\'s moduli of materials to merge do not match')
                if not mat.rho == dup_mat.rho: raise Exception('Atom densities of materials to merge do not match')

        xsection_ids = []
        for xsec in self.xsections + structure_2.xsections:
            # Add cross section to list of new structure's cross sections if ID is not already taken
            if not xsec.id in xsection_ids:
                new_structure_xsections.append(xsec)
                xsection_ids.append(xsec.id)
            else:
                # Verify that cross sections to merge are identical
                dup_xsec = next(filter(lambda new_xsec: new_xsec.id == xsec.id, new_structure_xsections), None)
                if not xsec.A == dup_node.A: raise Exception('Area of xsections to merge do not match')
                if not xsec.Iy == dup_node.Iy: raise Exception('Iy of xsections to merge do not match')
                if not xsec.Iz == dup_node.Iz: raise Exception('Iz of xsections to merge do not match')
                if not xsec.Ip == dup_node.Ip: raise Exception('Ip of xsections to merge do not match')
                if not xsec.J == dup_node.J: raise Exception('J of xsections to merge do not match')

        for el in self.elements + structure_2.elements:
            # Check if element already exists between specified nodes
            duplicate_element = False
            for existing_el in new_structure_elements:
                node_pair = [existing_el.node_a.id, existing_el.node_b.id]
                if el.node_a.id in node_pair and el.node_b.id in node_pair:
                    duplicate_element = True
                    dup_element = existing_el
                    break
            # Add element to structure if not overlapping an already-assigned element
            if not duplicate_element:
                new_structure_elements.append(el)
            else:
                # Verify that elements to merge are identical
                if not el.material.name == dup_element.material.name: raise Exception('Materials of elements to merge do not match')
                if not el.xsec.id == dup_element.xsec.id: raise Exception('Cross sections of elements to merge do not match')
                for i_eq in range(3): 
                    if not el.para_eq[i_eq] is dup_element.para_eq[i_eq]: raise Exception('Parametric equations of elements to merge do not match')
                if not all(el.z_vec == dup_element.z_vec): raise Exception('zvec of elements to merge do not match')

        combined_structure = Structure(
            node_list = new_structure_nodes,
            material_list = new_structure_materials,
            xsection_list = new_structure_xsections,
            element_list = new_structure_elements
        )
        return combined_structure

    def __radd__(self, structure_2):
        new_structure = self.__add__(structure_2)
        return new_structure

    def __iadd__(self, structure_2):
        self = self + structure_2
        return self

    def add_nodes(self, node_list):
        if isinstance(node_list, Node): node_list = [node_list]
        if not isinstance(node_list, list): raise Exception('node_list type error: must be a list or a node')
        node_ids = [nd.id for nd in self.nodes]
        for nd in node_list:
            if isinstance(nd, Node) and not nd.id in node_ids:
                self.nodes.append(nd)
                node_ids.append(nd.id)
            else:
                if len(nd) > 0:
                    coords = np.array(nd)
                    if len(coords) < 6: coords = np.append(coords, np.zeros(6 - len(coords)))
                    new_node = Node(self, coords)
                else:
                    new_node = Node(self)
                self.nodes.append(new_node)
                node_ids.append(new_node.id)

    def add_constraints(self, constraints: list):
        for constraint in constraints:
            con = np.array(constraint[1:-1])
            nd = next(filter(lambda nd: nd.id == constraint[0], self.nodes), None)
            if nd is None: raise Exception('No matching node id in structure for constraint')
            else:
                # Fill con array with ones if only a few dimensions are given
                if len(con) < 6: con = np.append(con, np.ones(6 - len(con)))
                nd.fixed_dof = con

    def add_materials(self, material_list):
        if isinstance(material_list, material): material_list = [material_list]
        if not isinstance(material_list, list): raise Exception('material_list type error: must be a list or material')
        # Create materials with provided data; Add to 'materials' list
        material_names = [existing_mat.name for existing_mat in self.materials]
        for mat in material_list:
            if isinstance(mat, material) and not mat.name in material_names:
                self.materials.append(mat)
                material_names.append(mat.name)
            elif isinstance(mat, list) and not mat[0] in material_names:
                new_material = Material(*tuple(mat))
                self.materials.append(new_material)
                material_names.append(new_material.name)

    def add_xsections(self, xsection_list):
        if isinstance(xsection_list, xsection): xsection_list = [xsection_list]
        if not isinstance(xsection_list, list): raise Exception('xsection_list type error: must be a list or xsection')
        xsection_ids = [xsec.id for xsec in self.xsections]
        for xsec in xsection_list:
            if isinstance(xsec, XSection) and not xsec.id in xsection_ids:
                self.xsections.append(xsec)
                xsection_ids.append(xsec.id)
            elif isinstance(xsec, list) and not xsec[0] in xsection_ids:
                new_xsection = XSection(*tuple(xsec))
                self.xsections.append(new_xsection)
                xsection_ids.append(new_xsection.id)

    def add_elements(self, element_list):
        if isinstance(element_list, Element): element_list = [element_list]
        if not isinstance(element_list, list): raise Exception('element_list type error: must be a list or element')
        for el in element_list:
            # Get node ids of element to add
            if isinstance(el[0], Node): node_a = el[0].id
            else: node_a = el[0]
            if isinstance(el[1], Node): node_b = el[1].id
            else: node_b = el[1]

            # Check if element already exists between specified nodes - create and add if not
            duplicate_element = False
            for existing_el in self.elements:
                node_pair = [existing_el.node_a.id, existing_el.node_b.id]
                if node_a in node_pair and node_b in node_pair:
                    duplicate_element = True
                    break
            if not duplicate_element:
                new_element = Element(*tuple([self] + el))
                self.elements.append(new_element)

    def add_node_bonds(self, element_pair: list, bond_type: str, bond_parameters: list):
        # If element_pair is specified as a node id triplet, find matching pair of elements 
        if len(element_pair) == 3:
            el_a_nodes = [element_pair[0], element_pair[1]]
            el_b_nodes = [element_pair[1], element_pair[2]]
            element_a = [el for el in self.elements if set(el_a_nodes) == set([el.node_a.id, el.node_b.id])]
            element_b = [el for el in self.elements if set(el_b_nodes) == set([el.node_a.id, el.node_b.id])]
            if len(element_a) == 0 or len(element_b) == 0: raise Exception('No element exists between specified nodes')
            element_pair = element_a + element_b
        elif len(element_pair) != 2: raise Exception('Only two elements can be bonded at a time')

        # Find node connecting the pair of elements
        connecting_node = next(filter(lambda nd: all([el in nd.elements for el in element_pair]), self.nodes), None)
        if connecting_node is None: raise Exception('No connecting node found for element pair')

        # Build list of needed number of atoms
        if not bond_type in bond.N_atoms: raise Exception('Bond type not listed in N_atom dictionary')
        else: N_atoms = bond.N_atoms[bond_type]

        # Join element end atoms with specified bonds as if the two were one element
        for i_bnd in range(N_atoms - 2):
            if element_pair[0].node_a is connecting_node:
                el_a_atoms = element_pair[0].atoms[(N_atoms - i_bnd - 3):None:-1]
            else:
                el_a_atoms = element_pair[0].atoms[-(N_atoms - i_bnd - 2):]
            if element_pair[1].node_a is connecting_node:
                el_b_atoms = element_pair[1].atoms[:(i_bnd + 1)]
            else:
                el_b_atoms = element_pair[1].atoms[-1:(-i_bnd - 2):-1]
            atom_set = el_a_atoms + [connecting_node.atom] + el_b_atoms
            atom_ids = [atm.id for atm in atom_set]
            connecting_node.create_bond(atom_ids, bond_type, bond_parameters)

    def rotate(self, rotation_angle: float, axis: list | np.ndarray | None = None):
        '''
        A function to return a copy of the structure rotated about a specified axis.
        Arguments:
          - axis: list | np.array = [1, 0, 0] - The axis of rotation specified by:
              1. x,y,z coordinates of a vector starting at the origin
              2. a pair of nodes
        '''
        if axis is None: axis = [1, 0, 0]
        if len(axis) < 1 or len(axis) > 3: raise Exception('Invalid specified axis of rotation')
        if all([isinstance(coord, (int, float)) for coord in axis]):
            if len(axis) < 3: axis = np.concat(np.array(axis), np.zeros(3 - len(axis)))
            rotation_axis = axis
        elif all([isinstance(node_obj, node) for node_obj in axis]):
            if len(axis) != 2: raise Exception('Only two nodes can be specified to determine the axis of rotation')
            rotation_axis = axis[1].coords[0:2] - axis[0].coords[0:2]
        #### Define transformation matrix
        euler_angles = [0, 0, 0]
        transform_matrix = rotation_axis
        return NotImplemented #self.transform_affine(transform_matrix)

    def mirror(self, mirror_plane_pts: np.ndarray):
        ########### Add code to construct transformation matrix 
        return NotImplemented #self.transform_affine(transform_matrix)

    def transform_affine(self, transform_matrix: np.ndarray):
        new_structure = cp.deepcopy(self)
        for i, nd in enumerate(self.nodes):
            transformed_coords = affine_matrix @ np.concat(nd.coords, np.array([1]))
            new_structure.nodes[i].coords = transformed_coords[0:2]
        return NotImplemented #new_structure
    
    def discretize(self, atom_distance: float, start_atom_id: int = 1, start_bond_id: dict | None = None) -> tuple:
        '''
        A function to discretize a structure given a specified atom spacing using the element.discretize() method.
        It returns data of beams, bonds joining beams to the nodes, and angles between beams joined at the nodes,
        all formatted for straightforward use with the Simulation class' functions.

        Arguments:
         - atom_distance: float - the maximum distance permitted between atoms
         - start_atom_id: int = 0 - the number of atoms already in the simulation (offsets the atom ids)

        Outputs:
         - A tuple of data for beam, bond, and angle creation (atoms, bonds, angles, dihedrals)

        '''
        # Reset atoms and bonds
        self.atoms = []
        self.bonds = []
        self.atom_type_list = []
        self.bond_type_list = []
        self.start_atom_id = start_atom_id
        if start_bond_id is None: start_bond_id = {}
        self.start_bond_id = start_bond_id

        # Discretize the structure's nodes and elements into atoms
        for nd in self.nodes: nd.discretize()
        for el in self.elements: el.discretize(atom_distance)
        
        # Bond element pairs through nodes (ie. with angle, dihedral, etc. bonds)
        if not self.element_connections is None:
            for el_connection in self.element_connections:
                self.add_node_bonds(*tuple(el_connection))

        return (self.atom_type_list, self.bond_type_list, self.atoms, self.bonds)

class Node:
    '''
    An object representing an endpoint or connection point between elements.

    Attributes:
        parent_structure: structure - 
        id: int - A unique ID (within the structure)
        coords: np.array (1 x 6) - 
        fixed_dof: np.array (1 x 6) = [0,0,0,0,0,0] - 
        atom: atom - 

    Methods:
        __init__ - 

        assign_connected_elements() - 
    '''
    def __init__(self, parent_structure: Structure, coords: np.ndarray | None = None, constraint: np.ndarray | None = None):
        if constraint is None: constraint = np.array([0,0,0,0,0,0])
        self.parent_structure = parent_structure
        self.id: int = len(parent_structure.nodes)
        self.coords = coords
        self.fixed_dof = constraint
        self.atom = None
        self.bonds = []

    def __str__(self):
        print_str = f'ID {self.id} - coords ({self.coords}), Fixed DOF {self.fixed_dof}'
        if not self.atom is None: print_str += f', atom ID {self.atom.id}'
        if len(self.bonds) > 0:
            print_str += '\n'
            for bnd in self.bonds: print_str += f'    {bnd}\n'
        return print_str

    def assign_connected_elements(self):
        '''
        A function to list all elements connected to the node.
        This is important for discretization in allowing for the removal of coincident atoms at connected beam ends.
        '''
        self.elements = [el for el in self.parent_structure.elements if self in [el.node_a, el.node_b]]

    def edit_coords(self, new_coords: np.array):
        if len(new_coords) < 6: new_coords = np.append(new_coords, np.zeros(6 - len(new_coords)))
        self.coords = new_coords
        for el in self.elements:
            if el.para_linear is True: para_eq = []
            else: para_eq = el.para_eq
            el.set_parametric_eqs(para_eq)
            self.length = calc_arc_length([self.node_a.coords, self.node_b.coords], para_eq, is_linear=self.para_linear)

    def create_atom(self, coords:np.ndarray):
        return Atom(coords, parent_node=self)

    def create_bond(self, atom_ids: list, bond_type: str, parameters: list):
        return Bond(atom_ids, bond_type, parameters, parent_node=self)
    
    def discretize(self):
        return self.create_atom(self.coords)

class XSection:
    '''
    An object that contains cross-section data, for convenient assigning to various elements

    Attributes:
        id: int - Unique ID
        atom_diameter: float - diameter of atoms used in discretization
        thickness_coefficient: float = 1 - 
        stretching_coefficient: float = 1
        bending_coefficient: float = 1

    Methods:
        __init__ - Initializes an instance with material and cross section properties
        __str__ - generates a string representing an instance's data for easy printing
    '''
    def __init__(
        self,
        id_num: int,
        atom_diameter: float,
        *,
        thickness_coefficient: float = 1,
        stretching_coefficient: float = 1,
        bending_coefficient: float = 1
        ):
        self.id = id_num
        self.atom_diameter = atom_diameter
        self.thickness_coefficient = thickness_coefficient
        self.stretching_coefficient = stretching_coefficient
        self.bending_coefficient = bending_coefficient

    def __str__(self):
        xsec_str_1 = f'ID {self.id} - atom diameter = {self.atom_diameter}, '
        xsec_str_2 = f'Coefficients: thickness = {self.thickness_coefficient}, stretching = {self.stretching_coefficient}, bending = {self.bending_coefficient}'
        return xsec_str_1 + xsec_str_2

class Material:
    '''
    An object containing material data, for convenient assigning to various elements

    Attributes:
    name: str - A unique name
    E: float - Young's Modulus
    rho: float - Density of atoms used in discretization
    '''
    def __init__(self, name: str, E: float, rho: float):
        self.name = name
        self.E = E
        self.rho = rho

    def __str__(self):
        return f'Name {self.name} - (E, rho)=({self.E}, {self.rho})'

class Element:
    '''
    An object representing a parametric member extending between two nodes.

    Attributes:
        parent_structure: structure - The structure containing the element
        node_a: node - A node at one end of the element
        node_b: node - A node at the other end of the element
        material: str
        xsec: int - ID of cross section to apply to element
        para_eq: list = [] - An optional list (1 x 3) of parametric equations defining the x,y,z values along the element (with parameter between 0 at node_a and 1 at node_b)
        z_vec = [] - A list or numpy array representing a vector defining the orientation of the element's cross section

    Methods:
        __init__ - 
        __str__ - 

        set_parametric_eqs - Set parametric equations defining element path

        set_material - Assign new or existing material to element

        set_xsection - Assign new or existing xsection to element

        discretize() - Discretize element into atoms separated by no more than a provided distance
    '''
    def __init__(
        self,
        parent_structure: Structure,
        node_a: int | Node,
        node_b: int | Node,
        mat: list | Material | str | None = None,
        xsec: list | XSection | int | None = None,
        para_eq: list | None = None
        ):

        self.parent_structure = parent_structure
        self.atoms = []
        self.bonds = []

        # Assign end nodes
        if isinstance(node_a, int): 
            matching_node_a = next(filter(lambda nd: nd.id == node_a, parent_structure.nodes), None)
            if matching_node_a is None: raise Exception('No matching node found for element end a')
            self.node_a = matching_node_a
        else: 
            self.node_a = node_a
        if isinstance(node_b, int): 
            matching_node_b = next(filter(lambda nd: nd.id == node_b, parent_structure.nodes), None)
            if matching_node_b is None: raise Exception('No matching node found for element end b')
            self.node_b = matching_node_b
        else:
            self.node_b = node_b

        # Set parametric equations for element path (if provided) and calculate element length
        if para_eq is None: para_eq = []
        self.set_parametric_eqs(para_eq)
        self.length = calc_arc_length([self.node_a.coords, self.node_b.coords], para_eq=para_eq, is_linear=self.para_linear)

        # Assign material and cross section - create instance if info is provided instead of objects
        # If not specified and the structure only has one, assign that one as the default
        if not mat is None: self.set_material(mat)
        elif len(self.parent_structure.materials) == 1: self.set_material(self.parent_structure.materials[0])
        if not xsec is None: self.set_xsection(xsec)
        elif len(self.parent_structure.xsections) == 1: self.set_xsection(self.parent_structure.xsections[0])

    def __str__(self):
        print_str = f'Node pair ({self.node_a.id}, {self.node_b.id}) - material {self.material.name}, x-section {self.xsec.id}'
        print_str += f', length {self.length}, Is_Linear {self.para_linear}'
        print_element_atoms_bonds = False
        if print_element_atoms_bonds is True:
            if len(self.atoms) > 0:
                print_str += f'\n    atom IDs:\n [{self.atoms[0].id}'
                for atm in self.atoms: print_str += f', {atm.id}'
                print_str += ']'
            if len(self.bonds) > 0:
                print_str += '\n'
                for bnd in self.bonds: print_str += f'    {bnd}\n'
        return print_str

    def set_parametric_eqs(self, para_eq: list):
        # Handle parametric equation inputs - create linear parametric equations if none are provided
        if len(para_eq) > 0:
            if len(para_eq) < 3: para_eq.append([0 for i in range(3-len(para_eq))])

            # Replace any constants with callable functions
            for coord_eq in para_eq: 
                if not callable(coord_eq):
                    coord_eq = lambda t: coord_eq

            # Set node coordinate given equation and other node if not set
            if self.node_a.coords is None and not self.node_b.coords is None:
                coords = np.array([para_eq[0](1), para_eq[1](1), para_eq[2](1), 0, 0, 0])
                self.node_a.coords = self.node_b.coords - coords
            elif self.node_b.coords is None and not self.node_a.coords is None:
                coords = np.array([para_eq[0](1), para_eq[1](1), para_eq[2](1), 0, 0, 0])
                self.node_b.coords = self.node_a.coords + coords

            # Check all coordinates for mismatching values with parametric function
            for i_eq, coord_eq in enumerate(para_eq):
                if coord_eq(0) != 0: 
                    raise Exception(f'Parametric equation must satisfy f(0) = 0 (positions must match at node_a)')
                if abs(coord_eq(1) + self.node_a.coords[i_eq] - self.node_b.coords[i_eq]) > 10**-6:
                    raise Exception(f'Parametric equation f(1)={coord_eq(1)} does not coincide with end node with ID {self.node_b.id}')
            self.para_eq = para_eq
            self.para_linear: bool = False
        elif len(self.node_a.coords) > 0 and len(self.node_b.coords) > 0:
            x_eq = lambda t: t * (self.node_b.coords[0] - self.node_a.coords[0])
            y_eq = lambda t: t * (self.node_b.coords[1] - self.node_a.coords[1])
            z_eq = lambda t: t * (self.node_b.coords[2] - self.node_a.coords[2])
            self.para_eq = [x_eq, y_eq, z_eq]
            self.para_linear: bool = True
        else: raise Exception('Could not parameterize - No parametric equations were provided and at least one end node has an undefined position')

    def set_material(self, mat: list | Material | str):
        if isinstance(mat, list):
            self.parent_structure.add_materials([mat])
            self.material = next(filter(lambda mat_entry: mat_entry.name == mat[0], self.parent_structure.materials), None)
        elif isinstance(mat, Material):
            self.parent_structure.add_materials([mat])
            self.material = next(filter(lambda mat_entry: mat_entry.name == mat.name, self.parent_structure.materials), None)
        else:
            mat = next(filter(lambda mat_entry: mat_entry.name == mat, self.parent_structure.materials), None)
            if mat is None: raise Exception('No material with matching name assigned to structure')
            else: self.material = mat

    def set_xsection(self, xsec: list | XSection | int):
        if isinstance(xsec, list):
            self.parent_structure.add_xsections([xsec])
            self.xsec = next(filter(lambda xsec_entry: xsec_entry.id == xsec[0], self.parent_structure.xsections), None)
        elif isinstance(xsec, XSection):
            self.parent_structure.add_xsections([xsec])
            self.xsec = next(filter(lambda xsec_entry: xsec_entry.id == xsec.id, self.parent_structure.xsections), None)
        else:
            structure_xsec = next(filter(lambda xsec_entry: xsec_entry.id == xsec, self.parent_structure.xsections), None)
            if structure_xsec is None: raise Exception('No xsection with matching id assigned to structure')
            else: self.xsec = structure_xsec

    def create_atom(self, coords:np.ndarray):
        return Atom(coords, parent_element=self)

    def create_bond(self, atom_ids: list, bond_type: str, parameters: list):
        return Bond(atom_ids, bond_type, parameters, parent_element=self)

    def discretize(self, atom_distance: float) -> tuple:
        '''
        A function to discretize an element with spacing not greater than the specified atom distance.
        It removes overlapping atoms at nodes where elements join, and returns the list of atoms.
        '''
        ### Do not call except from structure.discretize() -- atom ids need to be fully reassigned
        # Compute atom coordinates with parametric equations
        N_bonds = int(np.ceil(self.length/atom_distance))
        if N_bonds < 1: raise Exception('There must be at least 2 atoms per element')
        t = np.linspace(0, 1, N_bonds + 1)
        (x_eq, y_eq, z_eq) = tuple(self.para_eq)
        x_vec = x_eq(t) + self.node_a.coords[0]
        y_vec = y_eq(t) + self.node_a.coords[1]
        z_vec = z_eq(t) + self.node_a.coords[2]
        atom_coords = np.array([x_vec, y_vec, z_vec, t*0, t*0, t*0]).T
        ## TODO - interpolate twist/moment to set atom angles (instead of 0, 0, 0)

        # Remove end atom coordinates (overlapping with node atoms)
        atom_coords = atom_coords[1:-1]

        # Create the element's atoms
        for coords in atom_coords: self.create_atom(coords)

        # Create the element's two-atom bonds
        rest_length = self.length / N_bonds
        bond_stiffness = self.material.E * self.xsec.atom_diameter**2 / (2*rest_length)
        for at in self.atoms[0:-1]: self.create_bond([at.id, at.id + 1], 'bond', [bond_stiffness, rest_length])
        self.node_a.create_bond([self.node_a.atom.id, self.atoms[0].id], 'bond', [bond_stiffness, rest_length])
        self.node_b.create_bond([self.atoms[-1].id, self.node_b.atom.id], 'bond', [bond_stiffness, rest_length])

        # Create the element's four-atom bonds (dihedrals)
        angular_stiffness = self.material.E * self.xsec.atom_diameter**4 / (12*rest_length)
        atom_list = [self.node_a.atom] + self.atoms + [self.node_b.atom]
        atom_list_ids = [atm.id for atm in atom_list]
        if len(atom_list) >= 4:
            dvecs = [(atom_list[i+1].coords[0:3] - atom_list[i].coords[0:3]) / rest_length for i in range(N_bonds)]
            calc_vec_angle = lambda vec_a, vec_b: np.atan2(np.linalg.norm(np.cross(vec_a, vec_b)), np.dot(vec_a, vec_b))
            plane_angles = [calc_vec_angle(dvecs[i_angle], dvecs[i_angle+1]) for i_angle in range(N_bonds - 1)]

            # Calculate dihedral angles (angle between half-planes formed by each subsequent group of three nodes)
            dihedral_angles = []
            for i_dh_angle in range(N_bonds - 2):
                (vec_1, vec_2, vec_3) = tuple(dvecs[i_dh_angle:i_dh_angle + 3])
                #(vec_a, vec_b) = (np.cross(vec_1, vec_2), np.cross(vec_2, vec_3))
                #(vec_a, vec_b) = (vec_a / np.linalg.norm(vec_a), vec_b / np.linalg.norm(vec_b))
                dh_angle_y = np.linalg.norm(vec_2)*np.dot(vec_1, np.cross(vec_2, vec_3))
                dh_angle_x = np.dot(np.cross(vec_1, vec_2), np.cross(vec_2, vec_3))
                dihedral_angles.append(np.atan2(dh_angle_y, dh_angle_x))

            # Create node and element dihedral bonds
            rest_angles = [dihedral_angles[0], plane_angles[0], plane_angles[1]]
            ## TODO - find appropriate coefficient for dihedral stiffness (instead of just 'angular_stiffness')
            dihedral_stiffness = angular_stiffness * np.sin(rest_angles[1]) * np.sin(rest_angles[2])
            dihedral_parameters = [3, dihedral_stiffness, 1, rest_angles[0], 1, 1, 90, 0, 1, 90, 0,
                                      angular_stiffness, 0, 0, 0, 1, rest_angles[1], 1, 0, 0, 0,
                                      angular_stiffness, 0, 0, 0, 0, 0, 0, 1, rest_angles[2], 1]
            self.node_a.create_bond(atom_list_ids[:4], 'dihedral', dihedral_parameters)

            for i_at, at in enumerate(self.atoms[0:-4]):
                rest_angles = [plane_angles[i_at + 1], plane_angles[i_at + 2], dihedral_angles[i_at + 1]]
                dihedral_stiffness = 1 * np.sin(rest_angles[1]) * np.sin(rest_angles[2])
                dihedral_parameters = [3, dihedral_stiffness, 1, rest_angles[0], 1, 1, 90, 0, 1, 90, 0,
                                          angular_stiffness, 0, 0, 0, 1, rest_angles[1], 1, 0, 0, 0,
                                          angular_stiffness, 0, 0, 0, 0, 0, 0, 1, rest_angles[2], 1]
                self.create_bond([at.id + 1, at.id + 2, at.id + 3, at.id + 4], 'dihedral', dihedral_parameters)

            rest_angles = [plane_angles[-2], plane_angles[-1], dihedral_angles[-1]]
            dihedral_stiffness = 1 * np.sin(rest_angles[1]) * np.sin(rest_angles[2])
            dihedral_parameters = [3, dihedral_stiffness, 1, rest_angles[0], 1, 1, 90, 0, 1, 90, 0,
                                      angular_stiffness, 0, 0, 0, 1, rest_angles[1], 1, 0, 0, 0,
                                      angular_stiffness, 0, 0, 0, 0, 0, 0, 1, rest_angles[2], 1]
            self.node_b.create_bond(atom_list_ids[-4:], 'dihedral', dihedral_parameters)
            
        return (self.atoms, self.bonds)

class AtomType:
    def __init__(self, parent_structure: Structure, diameter: float, density: float):
        self.id = len(parent_structure.atom_types) + parent_structure.start_atom_type_id
        self.diameter = diameter
        self.density = density

    def __str__(self):
        print_str = f'Atom Type ID {self.id} - diameter {self.diameter}, density {self.density}'
        return print_str

## TODO - refactor to pass in atom_type ID or diameter, density to assign .type_id (or to generate new atom type)
class Atom:
    def __init__(
        self,
        coords: np.ndarray,
        atom_type: int | AtomType,
        *,
        N_previous_atoms: int = 0,
        parent_element: Element | None = None,
        parent_node: Node | None = None,
        parent_structure: Structure | None
        ):

        self.coords = coords
        self.parent_element = parent_element
        self.parent_node = parent_node
        self.type_id = None
        self.fixed_dof = [None, None, None]

        # Calculate and assign next available id
        if not parent_element is None and not parent_node is None: raise Exception('Atom cannot belong to both a node and an element')
        if not parent_element is None:
            diameter = parent_element.xsec.atom_diameter
            density = parent_element.material.rho
            parent_element.atoms.append(self)
            if not parent_structure is None and not parent_structure is parent_element.parent_structure:
                raise Exception('Parent structure of parent element does not match parent structure provided (parent structure need not be provided if a parent element is given)')
            else:
                parent_structure = parent_element.parent_structure
        if not parent_node is None:
            ## TODO - refactor to handle node atom properties more predictably
            diameter = parent_node.elements[0].xsec.atom_diameter
            density = parent_node.elements[0].material.rho
            for i_dof in range(3): 
                if parent_node.fixed_dof[i_dof] == 1:
                    self.fixed_dof[i_dof] = 0
            parent_node.atom = self
            if not parent_structure is None and not parent_structure is parent_node.parent_structure:
                raise Exception('Parent structure of parent node does not match parent structure provided (parent structure need not be provided if a parent node is given)')
            else:
                parent_structure = parent_node.parent_structure
        if not parent_structure is None:
            N_previous_atoms = parent_structure.start_atom_id + len(parent_structure.atoms)

        #################################
        if parent_structure is None and not atom_type is None: raise Exception('A parent structure is needed to assign atom types')
        if isinstance(atom_type, AtomType):
            self.atom_type = atom_type
        elif isinstance(atom_type, int):
            if parent_structure is None: raise Exception('Atom type cannot be referenced by ID when the atom has no parent structure assigned')
            matching_atom_type = next(filter(lambda type_entry: type_entry.id == atom_type, parent_structure.atom_types), None)
        #elif isinstance(atom_type, list):
            #if parent_struct

            # Calculate and assign next type id (distinct combination of type and parameters)
            for atm_type in parent_structure.atom_type_list:
                if diameter == atm_type[1] and density == atm_type[2]:
                    self.type_id = atm_type[0]
            if self.type_id is None:
                self.type_id = len(parent_structure.atom_type_list)
                parent_structure.atom_type_list.append([self.type_id, diameter, density])
            parent_structure.atoms.append(self)

        ## TODO - Handle atom_type's of atoms not belonging to node or element

        self.id = N_previous_atoms

## TODO - use structure.bond_type_list to reference type and properties - to remove attributes from bond object
class Bond:
    N_atoms: dict = {'bond': 2, 'angle': 3, 'dihedral': 4}
    def __init__(
        self,
        atom_ids: list,
        bond_type: str,
        parameters: list,
        *,
        N_previous_bonds: int = 0,
        parent_element: Element | None = None,
        parent_node: Node | None = None
        ):

        self.atom_ids = atom_ids
        self.type = bond_type
        self.parameters = parameters
        self.parent_element = parent_element
        self.parent_node = parent_node
        self.type_id = None

        # Calculate and assign next available id (by type) and type id
        if not parent_element is None and not parent_node is None: raise Exception('Only one parent object can be specified')
        parent_structure = None
        if not parent_element is None:
            parent_element.bonds.append(self)
            parent_structure = parent_element.parent_structure
        if not parent_node is None:
            parent_node.bonds.append(self)
            parent_structure = parent_node.parent_structure
        if not parent_structure is None:
            existing_bonds = [bnd for bnd in parent_structure.bonds if bnd.type == self.type]
            if not self.type in parent_structure.start_bond_id:
                parent_structure.start_bond_id[self.type] = 1
            N_previous_bonds = parent_structure.start_bond_id[self.type] + len(existing_bonds)

            # Calculate and assign next type id (distinct combination of type and parameters)
            for bnd_type in parent_structure.bond_type_list:
                parameters_match = compare_lists(self.parameters, bnd_type[2])
                if self.type == bnd_type[1] and parameters_match:
                    self.type_id = bnd_type[0]
            if self.type_id is None:
                self.type_id = 1 + len([bnd_typ for bnd_typ in parent_structure.bond_type_list if bnd_typ[1] == self.type])
                parent_structure.bond_type_list.append([self.type_id, self.type, self.parameters])
            parent_structure.bonds.append(self)

        self.id = N_previous_bonds
        if self.type_id == None: self.type_id = 1
        if len(self.atom_ids) != self.N_atoms[self.type]: raise Exception('Invalid number of atoms assigned to bond')

    def __str__(self):
        print_str = f'Bond ID {self.id} - type {self.type}, atoms {self.atom_ids}'
        #print_str = f', parameters {self.parameters}'
        return print_str

def calc_arc_length(coord_pair: list, is_linear: bool, para_eq: list[callable], bounds: list | None = None, n_pts: int = 200) -> float:
    '''
    Numerically calculate the arc length of a curve defined by parametric equations between the provided bounds.
    It uses the numpy gradient and trapz functions, and allows the number of points (n_pts) to be specified.

    Arguments:
        - para_eq: list[callable] (1 x 3) - [x_eq(), y_eq(), z_eq()] defining x, y, and z with single parameter
        - bounds: list = [0, 1] (1 x 2) - the start and end values of the parameter to integrate between
        - n_pts: int = 200 - the number of points used to evaluate the parametric equations for numeric integration
    
    Outputs:
        - length: float - The calculated arc-length between the start and end points along the curve.
    '''
    if is_linear is False:
        if bounds is None: bounds = [0, 1]
        t = np.linspace(bounds[0], bounds[1], n_pts)
        (x, y, z) = (para_eq[0](t), para_eq[1](t), para_eq[2](t))
        length = np.trapz(np.sqrt(np.gradient(x, t)**2 + np.gradient(y, t)**2 + np.gradient(z, t)**2), t)
    else:
        length = np.linalg.norm(coord_pair[1] - coord_pair[0])
    return length

def compare_lists(list_a: list, list_b: list) -> bool:
    '''
    Compare two lists; Return true if the lists are equal element-wise
    Runs recursively if elements of both lists are themselves lists
    '''
    lists_match = True
    if list_a is None or list_b is None:
        lists_match = False
    elif len(list_a) != len(list_b):
        lists_match = False
    else:
        for i, val_a in enumerate(list_a):
            val_b = list_b[i]
            if not type(val_a) is type(val_b):
                lists_match = False
            elif isinstance(val_a, list):
                lists_match = compare_lists(val_a, val_b)
            elif isinstance(val_a, np.ndarray):
                if not all(val_a == val_b):
                    lists_match = False
    return lists_match

if __name__ == "__main__":
    nodes = [[0,0], [0,1], [1,1], [1,0]]
    (E, rho) = (0.96 * 10**6, 0.5)
    materials = [['test_material_0', E, rho]]
    beam_thickness = 0.002
    xsecs = [[0, beam_thickness]]
    elements = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
    constraints = [[0, 1,1,1,1,1,1], [1, 1,1,1,1,1,1]]
    new_structure = structure(nodes, material_list = materials, xsection_list = xsecs, element_list = elements, constraint_list = constraints)
    new_structure.discretize(0.02)
    print(new_structure)