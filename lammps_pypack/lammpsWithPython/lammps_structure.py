import copy as cp
import numpy as np
from typing import Type
import warnings

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

## General TODO:
## bonds - Add bond_type object and refactor bond type handling
## structure, node, atom - Refactor constraints in a more lammps-friendly format
## element, structure - Add ability to offset discretization element lengths due to different node atom diameters
## structure - Test rotation, circular patterning, and mirroring methods

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

class Structure:
    '''
    An object representing a 3D structure, defined by nodes (points) connected by elements (lines or curves).
    Elements represent elastic members, and nodes joints between them.

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
        connection_list: list | None = None
        ):
        '''
        Arguments:
         - node_list - A list of tuples of node coordinates and diameters (optional) (# Nodes x # coords) or a list of nodes
         - constraint_list - A list of node coordinates to fix
           * in format (Node ID, DOF_1, DOF_2...) where DOF_1 - DOF_6 are 1 for fixed and 0 for free (default)
         - material_list - A list of materials to assign or of material properties to create materials
         - xsection_list - A list of cross sections to assign or of cross section properties to create cross sections
         - element_list - A list of elements specified by their end nodes
         - element_connections - A list of bonds used to join atoms from pairs of elements through their connecting node
           * by default, only 'single/bond' type bonds are used to connect element end atoms to node atoms
           * entry format: [[node_0, node_1, node_3], bond_type: str, bond_parameters: list]
        '''

        # Create materials, cross sections, and elements with provided data; Add to structure
        self.materials = []
        if not material_list is None: self.add_materials(material_list)
        self.xsections = []
        if not xsection_list is None: self.add_xsections(xsection_list)

        # Set orientation (Intrinsic Euler Angles); Create nodes and set fixed DOF
        self.orient = np.array([0, 0, 0])
        self.nodes = []
        if not node_list is None: self.add_nodes(node_list)
        if not constraint_list is None: self.add_constraints(constraint_list)

        # Create elements with provided element data; Add to 'elements' list; add to connected nodes
        self.elements = []
        if not element_list is None: self.add_elements(element_list)

        for nd in self.nodes: nd.assign_connected_elements()

        self.atoms = []
        self.bonds = []
        self.element_connections = []
        if not connection_list is None: self.add_connections(connection_list)

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
        
        # Combine materials
        new_structure_materials = cp.copy(self.materials)
        structure_2_materials = cp.copy(structure_2.materials)
        material_names = [mat.name for mat in new_structure_materials]
        for new_mat in structure_2_materials:
            # Add material to list of new structure's materials if its name is not already taken
            if not new_mat.name in material_names:
                new_structure_materials.append(mat)
                material_names.append(mat.name)
            else:
                # Verify that materials to merge are identical
                dup_mat = next(filter(lambda old_mat: old_mat.name == new_mat.name, new_structure_materials), None)
                if not new_mat.E == dup_mat.E: raise Exception('Young\'s moduli of materials to merge do not match')
                if not new_mat.rho == dup_mat.rho: raise Exception('Atom densities of materials to merge do not match')

        # Combine xsections
        new_structure_xsections = cp.copy(self.xsections)
        structure_2_xsections = cp.copy(structure_2.xsections)
        xsection_ids = [xsec.id for xsec in new_structure_xsections]
        change_xsec_ids = []
        for new_xsec in structure_2_xsections:
            # Add cross section to list of new structure's cross sections if ID is not already taken
            if not new_xsec.id in xsection_ids:
                new_structure_xsections.append(new_xsec)
                xsection_ids.append(new_xsec.id)
            else:
                # Verify that cross sections to merge are identical
                dup_xsec = next(filter(lambda old_xsec: old_xsec.id == new_xsec.id, new_structure_xsections), None)
                diameters_match = new_xsec.atom_diameter == dup_xsec.atom_diameter
                thickness_coeffs_match = new_xsec.thickness_coefficient == dup_xsec.thickness_coefficient
                stretching_coeffs_match = new_xsec.stretching_coefficient == dup_xsec.stretching_coefficient
                bending_coeffs_match = new_xsec.bending_coefficient == dup_xsec.bending_coefficient
                if not diameters_match or not thickness_coeffs_match or not stretching_coeffs_match or not bending_coeffs_match:
                    change_xsec_ids.append(new_xsec)

        # Reassign xsection ids to new xsections
        start_id = max(xsection_ids) + 1
        for i_xsec, xsec_to_change in enumerate(change_xsec_ids):
            xsec_to_change.id = start_id + i_xsec
            warnings.warn('XSections with duplicate ID given new ID')
        
        # Deepcopy structures to allow for ID reassigning without affecting input structures
        struct_1 = cp.deepcopy(self)
        struct_2 = cp.deepcopy(structure_2)

        # Combine nodes
        new_structure_nodes = cp.copy(struct_1.nodes)
        node_ids = [nd.id for nd in new_structure_nodes]
        change_node_ids = []
        for new_node in struct_2.nodes:
            matching_node = next(filter(lambda test_node: test_node.id == new_node.id, struct_1.nodes), None)
            overlapping_node = next(filter(lambda test_node: compare_lists(test_node.coords, new_node.coords), struct_1.nodes), None)
            if overlapping_node is None:
                if matching_node is None:
                    new_structure_nodes.append(new_node)
                    node_ids.append(new_node.id)
                else:
                    change_node_ids.append(new_node)
                    new_structure_nodes.append(new_node)
            else:
                if not matching_node is overlapping_node:
                    if new_node.diameter != overlapping_node.diameter: raise Exception('Diameters of coincident nodes to merge do not match')
                    if new_node.density != overlapping_node.density: raise Exception('Densities of coincident nodes to merge do not match')
                    new_node.change_id(overlapping_node.id, force=True)
                    warnings.warn('Overlapping nodes merged - check for node ID changes')
                    ## TODO - Check constraints

        # Reassign node ids to new nodes
        start_id = max(node_ids) + 1
        for i_nd, node_to_change in enumerate(change_node_ids):
            node_to_change.change_id(start_id + i_nd)
            warnings.warn('Node with duplicate ID given new ID')           

        # Combine elements
        new_structure_elements = cp.copy(struct_1.elements)
        for new_el in struct_2.elements:
            # Check if element already exists between specified nodes
            duplicate_element = False
            for existing_el in new_structure_elements:
                node_pair = [existing_el.node_a.id, existing_el.node_b.id]
                if new_el.node_a.id in node_pair and new_el.node_b.id in node_pair:
                    duplicate_element = True
                    dup_element = existing_el
                    break
            # Add element to structure if not overlapping an already-assigned element
            if not duplicate_element:
                new_structure_elements.append(new_el)
            else:
                # Verify that elements to merge are identical
                if not new_el.material.name == dup_element.material.name: raise Exception('Materials of elements to merge do not match')
                if not new_el.xsec.id == dup_element.xsec.id: raise Exception('Cross sections of elements to merge do not match')
                for i_eq in range(3):
                    if not new_el.para_eq[i_eq].__code__.co_code == dup_element.para_eq[i_eq].__code__.co_code: raise Exception('Parametric equations of elements to merge do not match')

        # Combine connections
        new_structure_connections = cp.copy(struct_1.element_connections)
        for new_connect in struct_2.element_connections:
            new_structure_connections.append(new_connect)
        
        # Tabulate node, element, and connection data
        new_structure_nodes.sort(key = lambda nd: nd.id)
        node_list = [(nd.coords, nd.diameter) for nd in new_structure_nodes]
        constraint_list = [[i_nd] + list(nd.fixed_dof) for i_nd, nd in enumerate(new_structure_nodes)]
        element_list = [[el.node_a.id, el.node_b.id, el.para_eq, el.material, el.xsec] for el in new_structure_elements]
        connection_list = [([nd.id for nd in connect[0]], connect[1], connect[2]) for connect in new_structure_connections]

        # Create combined structure
        combined_structure = Structure(
            node_list = node_list,
            material_list = new_structure_materials,
            xsection_list = new_structure_xsections,
            element_list = element_list,
            constraint_list = constraint_list,
            connection_list = connection_list
        )
        return combined_structure

    def __iadd__(self, structure_2):
        self = self + structure_2
        return self

    def add_nodes(self, node_list: list | tuple) -> None:
        if isinstance(node_list, tuple): node_list = [node_list]
        for node_data in node_list:
            # Parse node coordinates
            node_coords = node_data[0]
            if not node_coords is None:
                node_coords = np.array(node_coords)
                if np.size(node_coords) < 3:
                    np.append(node_coords, np.zeros(3 - np.size(node_coords)))
            
            # Parse node diameter
            if len(node_data) == 2:
                node_diameter = node_data[1]
            else:
                node_diameter = 0

            # Create node and add to structure's node list
            new_node = Node(self, coords = node_coords, diameter = node_diameter)
            self.nodes.append(new_node)
        return

    def add_constraints(self, constraints: list) -> None:
        for constraint in constraints:
            con = np.array(constraint[1:-1])
            nd = next(filter(lambda nd: nd.id == constraint[0], self.nodes), None)
            if nd is None: raise Exception('No matching node id in structure for constraint')
            else:
                # Fill con array with ones if only a few dimensions are given
                if len(con) < 6: con = np.append(con, np.ones(6 - len(con)))
                nd.fixed_dof = con
        return

    def add_materials(self, material_list: list | Material) -> None:
        if not isinstance(material_list, list): 
            material_list = [material_list]
        # Create materials with provided data; Add to 'materials' list
        material_names = [existing_mat.name for existing_mat in self.materials]
        for mat in material_list:
            if isinstance(mat, list) and not mat[0] in material_names:
                new_material = Material(*tuple(mat))
                self.materials.append(new_material)
                material_names.append(new_material.name)
            elif isinstance(mat, Material) and not mat.name in material_names:
                self.materials.append(mat)
                material_names.append(mat.name)
        return

    def add_xsections(self, xsection_list) -> None:
        if isinstance(xsection_list, XSection): xsection_list = [xsection_list]
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
        return

    def add_elements(self, element_list: list | tuple) -> None:
        if isinstance(element_list, tuple): element_list = [element_list]
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
                new_element = Element(*tuple([self] + list(el)))
                self.elements.append(new_element)
        return
    
    def add_connections(self, connection_list: list) -> None:
        for connect in connection_list:
            node_list = []
            for nd_id in connect[0]:
                node_list.append(next(filter(lambda nd: nd.id == nd_id, self.nodes), None))
            new_connection = (node_list, connect[1], connect[2])
            self.element_connections.append(new_connection)
        return

    def add_node_bonds(self, element_pair: list, bond_style: BondStyle | tuple, bond_parameters: list) -> None:
        # If element_pair is specified as a node id triplet, find matching pair of elements 
        if len(element_pair) == 3:
            if any([isinstance(nd, Node) for nd in element_pair]):
                el_a_nodes = [element_pair[0].id, element_pair[1].id]
                el_b_nodes = [element_pair[1].id, element_pair[2].id]
            else:
                el_a_nodes = [element_pair[0], element_pair[1]]
                el_b_nodes = [element_pair[1], element_pair[2]]
            element_a = [el for el in self.elements if set(el_a_nodes) == set([el.node_a.id, el.node_b.id])]
            element_b = [el for el in self.elements if set(el_b_nodes) == set([el.node_a.id, el.node_b.id])]
            if len(element_a) == 0 or len(element_b) == 0:
                raise Exception('No element exists between specified nodes')
            element_pair = element_a + element_b
            # Find node connecting the pair of elements
            connecting_node = next(filter(lambda nd: all([el in nd.elements for el in element_pair]), self.nodes), None)
            if connecting_node is None: raise Exception('No connecting node found for element pair')
        elif len(element_pair) != 2: raise Exception('Only two elements can be bound at a time')

        # Find matching bond style and needed number of atoms
        if isinstance(bond_style, tuple):
            if not isinstance(bond_style[0], str) or not isinstance(bond_style[1], str):
                raise Exception('Type and style of bond must be given as strings')
            for style in bond_style_list:
                if style.type == bond_style[0] and style.style == bond_style[1]:
                    bond_style = style
                    break
            if not isinstance(bond_style, BondStyle):
                raise Exception('No matching bond style found')
            N_atoms = bond_style.N_atoms

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
            connecting_node.create_bond(atom_ids, bond_style, bond_parameters)

        return

    def rotate(self, rotation_angle: float, axis: list | np.ndarray | None = None, copy: bool = True):
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
        r_mat = R.from_rotvec(rotation_angle*np.array(rotation_axis), degrees=True).as_matrix()
        full_orient = r_mat @ self.orient.as_matrix()
        self.orient = full_orient.as_euler('zxy', degrees=True)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = r_mat
        rotated_structure = self.transform_affine(rotation_matrix, copy)
        return rotated_structure

    def translate(self, offset: list | np.ndarray, copy: bool = True):
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 3] = offset
        translated_structure = self.transform_affine(transform_matrix, copy)
        return translated_structure

    def mirror(self, point_coords: np.ndarray, norm_vec: np.ndarray, copy: bool = True):
        self.translate(-point_coords, copy=False)
        mirror_matrix = np.eye(4)
        mirror_matrix[:3, :3] += -2*norm_vec @ norm_vec.T
        mirrored_structure = self.transform_affine(mirror_matrix, copy)
        self.translate(point_coords, copy=False)
        if copy: mirrored_structure.translate(point_coords, copy=False)
        return mirrored_structure

    def pattern_linear(self, axis: np.ndarray, n: int, offset: float | None, total_distance: float | None = None, connections: list | None = None):
        if total_distance is None and offset is None: raise Exception('Either offset or total distance must be specified')
        if not total_distance is None and not offset is None: raise Exception('Only one of offset or total distance can be specified')
        pattern = cp.deepcopy(self)
        if offset is None:
            offset = total_distance / (n + 1)
        for i_n in range(n):
            dist_offset = offset * (i_n + 1) * axis
            self += pattern.translate(dist_offset)
        if not connections is None: self.add_connections(connections)
        return self

    def pattern_circular(self, axis: np.ndarray, n: int, angle: float | None, total_angle: float | None = None, connections: list | None = None):
        if total_angle is None and angle is None: raise Exception('Either angle or total_angle must be specified')
        if not total_angle is None and not angle is None: raise Exception('Only one of angle or total angle can be specified')
        pattern = cp.deepcopy(self)
        if angle is None:
            angle = total_angle / (n + 1)
        for i_n in range(n):
            angle_offset = angle * (i_n + 1) * axis
            self += pattern.rotate(angle_offset, axis)
        if not connections is None: self.add_connections(connections)
        return self

    def transform_affine(self, affine_matrix: np.ndarray, copy: bool = True):
        if copy: new_structure = cp.deepcopy(self)
        else: new_structure = self
        for i, nd in enumerate(self.nodes):
            coord_vec = np.concat((nd.coords[0:3], np.ones(1)))
            transformed_coords = affine_matrix @ coord_vec
            new_structure.nodes[i].coords = transformed_coords[0:3]
        return new_structure
    
    def discretize(self, atom_distance: float, start_atom_id: int = 1, start_bond_id: int = 1) -> tuple:
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
        self.start_bond_id = start_bond_id

        # Discretize the structure's nodes and elements into atoms
        for nd in self.nodes: nd.discretize()
        for el in self.elements:
            el.discretize(atom_distance)
        
        # Bond element pairs through nodes (ie. with angle, dihedral, etc. bonds)
        if not self.element_connections is None:
            for el_connection in self.element_connections:
                self.add_node_bonds(*el_connection)
                
        return (self.atom_type_list, self.bond_type_list, self.atoms, self.bonds)

    def get_atom_id(self, node_pair: list, dist: float, percent: bool = True) -> int:
        if len(node_pair) != 2: raise Exception('Specify only two node IDs')
        element = None
        for existing_el in self.elements:
            el_node_pair = [existing_el.node_a.id, existing_el.node_b.id]
            if node_pair[0] in el_node_pair and node_pair[1] in el_node_pair:
                element = existing_el
                break
        if element is None: raise Exception('No matching element found between node IDs')

        # Calculate coordinates of point
        if not percent: dist = dist / element.length
        if not element.node_a.id == node_pair[0]: dist = 1 - dist
        x = element.node_a.coords[0] + element.para_eq[0](dist)
        y = element.node_a.coords[1] + element.para_eq[1](dist)
        z = element.node_a.coords[2] + element.para_eq[2](dist)
        point_coords = np.array([x, y, z])

        # Find nearest atom belonging to element
        least_dist = element.length
        for atom in element.atoms:
            check_dist = np.linalg.norm(atom.coords[0:3] - point_coords)
            if check_dist < least_dist:
                nearest_atom = atom
                least_dist = check_dist

        return nearest_atom.id

    def plot(self, fpath: str, fname: str):
        
        ax = plt.figure(figsize=(8, 6)).add_subplot(projection='3d')

        # Plot each node
        for nd in self.nodes:
            ax.plot(nd.coords[0], nd.coords[1], nd.coords[2], label='node', marker='o', color='black')
            ax.text(nd.coords[0], nd.coords[1], nd.coords[2], str(nd.id))

        # Plot each element
        for el in self.elements:
            t = np.linspace(0, 1, 100)
            x = el.node_a.coords[0] + el.para_eq[0](t)
            y = el.node_a.coords[1] + el.para_eq[1](t)
            z = el.node_a.coords[2] + el.para_eq[2](t)
            ax.plot(x, y, z, label='element', color='red')

        ax.axis('equal')
        # Save the plot image to the tutorials directory.
        img_fname = fpath + fname
        plt.savefig(img_fname)

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
    def __init__(
        self,
        parent_structure: Structure,
        coords: np.ndarray | None = None,
        constraint: np.ndarray | None = None,
        material = None,
        diameter: float = 0,
        density: float = 0
        ):

        if constraint is None: constraint = np.array([0,0,0,0,0,0])
        self.parent_structure = parent_structure
        self.id: int = len(parent_structure.nodes)
        self.coords = coords
        self.fixed_dof = constraint
        self.diameter = diameter
        self.density = density

        # Assign material - create instance if info is provided instead of objects
        # If not specified and the structure only has one, assign that one as the default
        if not material is None: self.set_material(material)
        elif len(self.parent_structure.materials) == 1: self.set_material(self.parent_structure.materials[0])
        else: self.material = None

        self.atom = None
        self.bonds = []

    def __str__(self):
        print_str = f'ID {self.id} - coords ({self.coords}), Fixed DOF {self.fixed_dof}'
        if not self.atom is None: print_str += f', atom ID {self.atom.id}'
        if len(self.bonds) > 0:
            print_str += '\n'
            for bnd in self.bonds: print_str += f'    {bnd}\n'
        return print_str

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

    def assign_connected_elements(self):
        '''
        A function to list all elements connected to the node.
        This is important for discretization in allowing for the removal of coincident atoms at connected beam ends.
        '''
        self.elements = [el for el in self.parent_structure.elements if self in [el.node_a, el.node_b]]

    def edit_coords(self, new_coords: np.array):
        if len(new_coords) < 3: new_coords = np.append(new_coords, np.zeros(3 - len(new_coords)))
        self.coords = new_coords
        for el in self.elements:
            if el.para_linear is True: para_eq = []
            else: para_eq = el.para_eq
            el.set_parametric_eqs(para_eq)
            self.length = calc_arc_length([self.node_a.coords, self.node_b.coords], para_eq, is_linear=self.para_linear)

    def change_id(self, new_id: int, force: bool = False):
        node_ids = [nd.id for nd in self.parent_structure.nodes]
        if not force:
            if new_id in node_ids:
                raise Exception('New node ID already taken')
        self.id = new_id

    def create_atom_type(self, parameters: list):
        return AtomType(parameters[0], parameters[1], self.parent_structure)

    def create_atom(self, coords:np.ndarray):
        return Atom(coords, parent_node=self)

    def create_bond(self, atom_ids: list, bond_style: str, parameters: list):
        return Bond(atom_ids, bond_style, parameters, parent_node=self)
    
    def discretize(self):
        new_atom = self.create_atom(self.coords)
        # Create a new atom type and assign to parent structure; ID to self
        new_atom_type = (self.id + 1, new_atom.diameter, new_atom.density)
        new_atom.type_id = self.id + 1
        self.parent_structure.atom_type_list.append(new_atom_type)
        return self

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
        node_a: int,
        node_b: int,
        para_eq: list | None = None,
        mat: list | Material | str | None = None,
        xsec: list | XSection | int | None = None,
        angular_rigidity_factor: float = 1,
        dihedral_rigidity_factor: float = 0.1
        ):

        self.parent_structure = parent_structure
        self.atoms = []
        self.bonds = []

        # Set stiffness scaling coefficients
        self.angular_rigidity_factor = angular_rigidity_factor
        self.dihedral_rigidity_factor = dihedral_rigidity_factor

        # Assign end nodes
        matching_node_a = next(filter(lambda nd: nd.id == node_a, parent_structure.nodes), None)
        if matching_node_a is None: raise Exception('No matching node found for element end a')
        self.node_a = matching_node_a
        matching_node_b = next(filter(lambda nd: nd.id == node_b, parent_structure.nodes), None)
        if matching_node_b is None: raise Exception('No matching node found for element end b')
        self.node_b = matching_node_b

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
                ref_coords = np.array([para_eq[0](1), para_eq[1](1), para_eq[2](1)])
                r_mat = R.from_euler('zyx', list(self.parent_structure.orient), degrees=True)
                coords =  r_mat.as_matrix() @ ref_coords
                self.node_a.coords = self.node_b.coords - coords
            elif self.node_b.coords is None and not self.node_a.coords is None:
                ref_coords = np.array([para_eq[0](1), para_eq[1](1), para_eq[2](1)])
                r_mat = R.from_euler('zyx', list(self.parent_structure.orient), degrees=True)
                coords =  r_mat.as_matrix() @ ref_coords
                self.node_b.coords = self.node_a.coords + coords

            # Check all coordinates for mismatching values with parametric function
            for i_eq, coord_eq in enumerate(para_eq):
                if coord_eq(0) != 0: 
                    raise Exception(f'Parametric equation must satisfy f(0) = 0 (positions must match at node_a)')
                if abs(coord_eq(1) + self.node_a.coords[i_eq] - self.node_b.coords[i_eq]) > 10**-6:
                    raise Exception(f'Parametric equation f(1)={coord_eq(1)} does not coincide with end node with ID {self.node_b.id}')
            self.para_eq = para_eq
            self.para_linear: bool = False
        elif not self.node_a.coords is None and not self.node_b.coords is None:
            x_eq = lambda t: t * (self.node_b.coords[0] - self.node_a.coords[0])
            y_eq = lambda t: t * (self.node_b.coords[1] - self.node_a.coords[1])
            z_eq = lambda t: t * (self.node_b.coords[2] - self.node_a.coords[2])
            self.para_eq = [x_eq, y_eq, z_eq]
            self.para_linear: bool = True
            self.dihedral_rigidity_factor = 0
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

    def create_bond(self, atom_ids: list, bond_style: tuple | BondStyle, parameters: list):
        return Bond(atom_ids, bond_style, parameters, parent_element=self)

    def discretize(self, atom_distance: float) -> tuple:
        '''
        A function to discretize an element with spacing not greater than the specified atom distance.
        It removes overlapping atoms at nodes where elements join, and returns the list of atoms.
        '''
        # Compute atom coordinates with parametric equations
        N_bonds = int(np.ceil(self.length/atom_distance))
        if N_bonds < 1: raise Exception('There must be at least 2 atoms per element')
        t = np.linspace(0, 1, N_bonds + 1)
        (x_eq, y_eq, z_eq) = tuple(self.para_eq)
        x_vec = x_eq(t) + self.node_a.coords[0]
        y_vec = y_eq(t) + self.node_a.coords[1]
        z_vec = z_eq(t) + self.node_a.coords[2]
        atom_coords = np.array([x_vec, y_vec, z_vec]).T

        # Remove end atom coordinates (overlapping with node atoms)
        atom_coords = atom_coords[1:-1]

        # Create the element's atoms
        for coords in atom_coords: self.create_atom(coords)

        # Calculate a new atom type ID
        max_node_id = max([nd.id for nd in self.parent_structure.nodes])
        i_el = self.parent_structure.elements.index(self)
        new_type_id = max_node_id + i_el + 2

        # Assign the new atom type to the parent structure; ID to self
        new_atom_type = (new_type_id, self.xsec.atom_diameter, self.material.rho)
        for atm in self.atoms:
            atm.type_id = new_type_id
        self.parent_structure.atom_type_list.append(new_atom_type)        

        # Create the element's two-atom bonds
        rest_length = self.length / N_bonds
        bond_stiffness = self.material.E * self.xsec.atom_diameter**2 / (2*rest_length)
        for at in self.atoms[0:-1]: self.create_bond([at.id, at.id + 1], ('bond', 'harmonic'), [bond_stiffness, rest_length])
        self.node_a.create_bond([self.node_a.atom.id, self.atoms[0].id], ('bond', 'harmonic'), [bond_stiffness, rest_length])
        self.node_b.create_bond([self.atoms[-1].id, self.node_b.atom.id], ('bond', 'harmonic'), [bond_stiffness, rest_length])

        # Create the element's four-atom bonds (dihedrals)
        angular_stiffness = self.angular_rigidity_factor * self.material.E * self.xsec.atom_diameter**4 / (12*rest_length)
        atom_list = [self.node_a.atom] + self.atoms + [self.node_b.atom]
        atom_list_ids = [atm.id for atm in atom_list]
        if len(atom_list) >= 4:
            dvecs = [(atom_list[i+1].coords[:] - atom_list[i].coords[:]) / rest_length for i in range(N_bonds)]
            calc_vec_angle = lambda vec_a, vec_b: 180 - (180/np.pi) * np.atan2(np.linalg.norm(np.cross(vec_a, vec_b)), np.dot(vec_a, vec_b))
            plane_angles = [calc_vec_angle(dvecs[i_angle], dvecs[i_angle+1]) for i_angle in range(N_bonds - 1)]

            # Calculate dihedral angles (angle between half-planes formed by each subsequent group of three nodes)
            dihedral_angles = []
            for i_dh_angle in range(N_bonds - 2):
                (vec_1, vec_2, vec_3) = tuple(dvecs[i_dh_angle:i_dh_angle + 3])
                dh_angle_y = np.linalg.norm(vec_2)*np.dot(vec_1, np.cross(vec_2, vec_3))
                dh_angle_x = np.dot(np.cross(vec_1, vec_2), np.cross(vec_2, vec_3))
                dihedral_angles.append(180/np.pi * np.atan2(dh_angle_y, dh_angle_x))

            # Create node and element dihedral bonds
            for i_atm, atm in enumerate(atom_list[0:-3]):
                rest_angles = [dihedral_angles[i_atm], plane_angles[i_atm], plane_angles[i_atm + 1]]
                dihedral_stiffness = self.dihedral_rigidity_factor * angular_stiffness * np.sin(rest_angles[1]) * np.sin(rest_angles[2])
                dihedral_parameters = [3, dihedral_stiffness, 1, rest_angles[0], 1, 1, 90, 0, 1, 90, 0,
                                          angular_stiffness/2, 0, 0, 0, 1, rest_angles[1], 1, 0, 0, 0,
                                          angular_stiffness/2, 0, 0, 0, 0, 0, 0, 1, rest_angles[2], 1]
                if not atm.parent_node is None:
                    atm.parent_node.create_bond(atom_list_ids[i_atm:i_atm + 4], ('dihedral','spherical'), dihedral_parameters)
                else:
                    self.create_bond(atom_list_ids[i_atm:i_atm + 4], ('dihedral','spherical'), dihedral_parameters)

        return (self.atoms, self.bonds)

class Atom:
    def __init__(
        self,
        coords: np.ndarray,
        *,
        N_previous_atoms: int = 0,
        parent_element: Element | None = None,
        parent_node: Node | None = None,
        parent_structure: Structure | None = None,
        diameter: float = 0,
        density: float = 0
        ):

        self.coords = coords
        self.parent_element = parent_element
        self.parent_node = parent_node
        self.fixed_dof = [None, None, None]

        # Verify correct parent object specification
        if not parent_element is None and not parent_node is None:
            raise Exception('Atom cannot belong to both a node and an element')
        if not parent_element is None:
            if not parent_structure is None and not parent_structure is parent_element.parent_structure:
                raise Exception('Parent structure of parent element does not match parent structure provided (parent structure need not be provided if a parent element is given)')
        elif not parent_node is None:
            if not parent_structure is None and not parent_structure is parent_node.parent_structure:
                raise Exception('Parent structure of parent node does not match parent structure provided (parent structure need not be provided if a parent node is given)')

        # Set diameter and density; Determine parent structure and link parents to child atom
        if not parent_element is None:
            parent_element.atoms.append(self)
            parent_structure = parent_element.parent_structure
            self.diameter = parent_element.xsec.atom_diameter
            self.density = parent_element.material.rho
        elif not parent_node is None:
            parent_node.atom = self
            parent_structure = parent_node.parent_structure
            self.diameter = parent_node.diameter
            self.density = parent_node.material.rho
            # Fill constraint array
            for i_dof in range(3):
                if parent_node.fixed_dof[i_dof] == 1:
                    self.fixed_dof[i_dof] = 0

        # Overwrite diameter and density if specified
        if diameter != 0:
            self.diameter = diameter
        if density != 0:
            self.density = density

        # Calculate and assign next available id
        if not parent_structure is None:
            N_previous_atoms = parent_structure.start_atom_id + len(parent_structure.atoms)
            parent_structure.atoms.append(self)
        self.id = N_previous_atoms

class Bond:
    N_atoms: dict = {'bond': 2, 'angle': 3, 'dihedral': 4}
    def __init__(
        self,
        atom_ids: list,
        bond_style: BondStyle | tuple,
        parameters: list | None = None,
        *,
        N_previous_bonds: int = 0,
        parent_element: Element | None = None,
        parent_node: Node | None = None,
        bond_type_list: list | None = None
        ):

        self.parent_element = parent_element
        self.parent_node = parent_node

        # Link bond to parent objects
        if not parent_element is None and not parent_node is None: 
            raise Exception('Only one parent object can be specified')
        parent_structure = None
        if not parent_element is None:
            parent_element.bonds.append(self)
            parent_structure = parent_element.parent_structure
        elif not parent_node is None:
            parent_node.bonds.append(self)
            parent_structure = parent_node.parent_structure

        # Determine number of bonds already in parent structure and assign id; Set bond_type_list
        if not parent_structure is None:
            N_previous_bonds = parent_structure.start_bond_id + len(parent_structure.bonds)
            parent_structure.bonds.append(self)
            bond_type_list = parent_structure.bond_type_list
        self.id = N_previous_bonds

        # Find and link matching bond style
        self.style = None
        if isinstance(bond_style, BondStyle):
            self.style = bond_style
        elif len(bond_style) != 2:
            raise Exception('Two (and only two) arguments determine bond style (type and style)')
        elif not isinstance(bond_style[0], str) or not isinstance(bond_style[1], str):
            raise Exception('Type and style of bond must be strings')
        else:
            for style in bond_style_list:
                if style.type == bond_style[0] and style.style == bond_style[1]:
                    self.style = style
                    break
            if self.style is None:
                raise Exception('No matching bond style found')

        # Verify the correct number of atoms were provided
        if not len(atom_ids) == self.style.N_atoms:
            raise Exception('Invalid number of atoms specified for bond type')
        self.atom_ids = atom_ids

        if parameters is None:
            if parent_structure is None:
                raise Exception('Bond parameters must be provided for bonds without parent structures')
            for existing_bond in parent_structure.bonds:
                if existing_bond.style == self.style and len(set(existing_bond.atom_ids) & set(self.atom_ids)) > 0:
                    parameters = existing_bond.parameters
                    break
            if parameters is None:
                raise Exception('No matching bond found - could not infer bond parameters')
        else:
            self.verify_parameters(parameters)
        self.parameters = parameters

        # Assign matching bond_type_id; append type to structure's bond type list if not
        if not bond_type_list is None:
            new_type = True
            for bond_type in bond_type_list:
                if bond_type[1] == self.style.type and bond_type[2] == self.style.style and compare_lists(bond_type[3], self.parameters):
                    self.type_id = bond_type[0]
                    new_type = False
            if new_type:
                self.type_id = len(bond_type_list) + 1
                bond_type_list.append((self.type_id, self.style.type, self.style.style, self.parameters))

    def __str__(self):
        print_str = f'Bond ID {self.id} - type/style {self.type}/{self.style}, atoms {self.atom_ids}'
        #print_str = f', parameters {self.parameters}'
        return print_str
    
    def verify_parameters(self, parameters: list):
        '''
        Function to compare number and type of parameters provided with the bond style parameter template
        Bond styles allowing multiple sets of parameters list a parameter N_sub_params followed by an int-type parameter
            - the int parameter provided specifies the number of repeated sub-parameter sets to expect
        '''
        # Build list of expected type of each parameter
        param_type_list = []
        i_param = 0
        for param in self.style.params:
            if param[0] == 'N_sub_params':
                if len(parameters) <= i_param:
                    raise Exception('Not enough parameters provided')
                if not isinstance(parameters[i_param], param[1]):
                    raise Exception('Invalid number of sub-parameter lines to include (must be int)')
                N_sub_params = parameters[i_param]
            if param[0] == 'sub_params':
                sub_params = param[1]
                for i_param_set in range(N_sub_params):
                    for sub_param in sub_params:
                        param_type_list.append(sub_param[1])
                        i_param = i_param + 1
                N_sub_params = 0
            else:
                param_type_list.append(param[1])
                i_param = i_param + 1
        
        # Verify number of provided parameters is correct
        if len(parameters) != len(param_type_list):
            raise Exception('Invalid number of provided parameters')

        # Verify type of each provided parameter is correct
        for i_param, param in enumerate(parameters):
            if not isinstance(param, param_type_list[i_param]):
                raise Exception('Invalid bond parameter type specified')
        
        return 0

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
            elif isinstance(val_a, float):
                if abs(val_a - val_b) > 10 ** -10:
                    lists_match = False
            elif not val_a == val_b:
                lists_match = False
    return lists_match