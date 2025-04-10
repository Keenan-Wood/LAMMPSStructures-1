import os
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import copy as cp

class frame:
    '''
    An object representing a 3D frame or truss.

    Attributes:
        nodes - list of node objects
        elements - list of element objects representing links between nodes
        materials - list of material objects storing material properties
        xsections - list of cross section objects storing cross section properties
        atoms - list of atom objects to use as lammps atoms
        atom_start_id - ID of first atom in frame

    Methods:
        __init__ - creates xsec and element objects as defined by input lists to initialize a frame instance
        __str__ - generates a string representing an instance's data for easy printing
        __add__ - 
        __radd__ -
        __iadd__ - 

        add_nodes - Create new nodes and add to frame

        add_constraints - Fix DOF of specified node DOF

        add_materials - Add new or existing materials to frame
        
        add_xsections - Add new or existing cross sections (xsection objects) to frame

        add_elements - Add new elements defined between nodes and add to frame

        discretize(atom_distance: float) - interpolates coordinates along elements based on provided atom_distance
          - returns data needed to add the frame geometry to a Simulation instance
    '''
    def __init__(
        self,
        node_list: list = [],
        *,
        constraint_list: list = [[]],
        material_list: list = [],
        xsection_list: list = [],
        element_list: list = []
        ):

        # Create nodes and set fixed DOF
        self.nodes = []
        if len(node_list) > 0: self.add_nodes(node_list)
        if len(constraint_list) > 0: self.add_constraints(constraint_list)

        # Create materials, cross sections, and elements with provided data; Add to frame
        self.materials = []
        if len(material_list) > 0: self.add_materials(material_list)
        self.xsections = []
        if len(xsection_list) > 0: self.add_xsections(xsection_list)

        # Create elements with provided element data; Add to 'elements' list; add to connected nodes
        self.elements = []
        if len(element_list) > 0: self.add_elements(element_list)

        for nd in self.nodes: nd.assign_connected_elements()
        self.atoms = []

    def __str__(self) -> str:
        print_str = f'Frame Data\n'
        print_str += f'Nodes:\n'
        print_str += f'{self.nodes}\n'
        print_str += f'X-Sections:\n'
        for xsec in self.xsections: print_str += f'{xsec}\n'
        print_str += f'Elements:\n'
        for el in self.elements: print_str += f'{el}\n'
        all_atoms = [at for at in el.atoms for el in self.elements]
        print_str += f'Number of Atoms: {len(all_atoms)}\n'
        return print_str

    def __add__(self, frame_2):
        if not isinstance(frame_2, frame): raise Exception('Only frame instances can be combined')
        new_frame_nodes = []
        new_frame_materials = []
        new_frame_xsections = []
        new_frame_elements = []

        node_ids = []
        for nd in self.nodes + frame_2.nodes:
            # Add node to list of new frame's nodes if ID is not already taken
            if not nd.id in node_ids:
                new_frame_nodes.append(nd)
                node_ids.append(nd.id)
            else:
                # Verify that nodes to merge are identical
                dup_node = next(filter(lambda new_node: new_node.id == nd.id, new_frame_nodes), None)
                if not all(nd.coords == dup_node.coords): raise Exception('Coordinates of nodes to merge do not match')
                if not all(nd.fixed_dof == dup_node.fixed_dof): raise Exception('Constraints of nodes to merge do not match')

        material_names = []
        for mat in self.materials + frame_2.materials:
            # Add material to list of new frame's materials if its name is not already taken
            if not mat.name in material_names:
                new_frame_materials.append(mat)
                material_ids.append(mat.name)
            else:
                # Verify that materials to merge are identical
                dup_mat = next(filter(lambda new_mat: new_mat.name == mat.name, new_frame_materials), None)
                if not mat.E == dup_mat.E: raise Exception('Young\'s moduli of materials to merge do not match')
                if not mat.rho == dup_mat.rho: raise Exception('Atom densities of materials to merge do not match')

        xsection_ids = []
        for xsec in self.xsections + frame_2.xsections:
            # Add cross section to list of new frame's cross sections if ID is not already taken
            if not xsec.id in xsection_ids:
                new_frame_xsections.append(xsec)
                xsection_ids.append(xsec.id)
            else:
                # Verify that cross sections to merge are identical
                dup_xsec = next(filter(lambda new_xsec: new_xsec.id == xsec.id, new_frame_xsections), None)
                if not xsec.A == dup_node.A: raise Exception('Area of xsections to merge do not match')
                if not xsec.Iy == dup_node.Iy: raise Exception('Iy of xsections to merge do not match')
                if not xsec.Iz == dup_node.Iz: raise Exception('Iz of xsections to merge do not match')
                if not xsec.Ip == dup_node.Ip: raise Exception('Ip of xsections to merge do not match')
                if not xsec.J == dup_node.J: raise Exception('J of xsections to merge do not match')

        for el in self.elements + frame_2.elements:
            # Check if element already exists between specified nodes
            duplicate_element = False
            for existing_el in new_frame_elements:
                node_pair = [existing_el.node_a.id, existing_el.node_b.id]
                if el.node_a.id in node_pair and el.node_b.id in node_pair:
                    duplicate_element = True
                    dup_element = existing_el
                    break
            # Add element to frame if not overlapping an already-assigned element
            if not duplicate_element:
                new_frame_elements.append(el)
            else:
                # Verify that elements to merge are identical
                if not el.material.name == dup_element.material.name: raise Exception('Materials of elements to merge do not match')
                if not el.xsec.id == dup_element.xsec.id: raise Exception('Cross sections of elements to merge do not match')
                for i_eq in range(3): 
                    if not el.para_eq[i_eq] is dup_element.para_eq[i_eq]: raise Exception('Parametric equations of elements to merge do not match')
                if not all(el.z_vec == dup_element.z_vec): raise Exception('zvec of elements to merge do not match')

        combined_frame = frame(
            node_list = new_frame_nodes,
            material_list = new_frame_materials,
            xsection_list = new_frame_xsections,
            element_list = new_frame_elements
        )
        return combined_frame

    def __radd__(self, frame_2):
        new_frame = self.__add__(frame_2)
        return new_frame

    def __iadd__(self, frame_2):
        self = self + frame_2
        return self

    def add_nodes(self, node_list: list):
        node_ids = [nd.id for nd in self.nodes]
        for nd in node_list:
            if isinstance(nd, node) and not nd.id in node_ids:
                self.nodes.append(nd)
                node_ids.append(nd.id)
            elif len(nd) > 0:
                coords = np.array(nd)
                if len(coords) < 6: coords = np.append(coords, np.zeros(6 - len(coords)))
                new_node = node(self, coords)
                self.nodes.append(new_node)
                node_ids.append(new_node.id)

    def add_constraints(self, constraints: list):
        for constraint in constraints:
            con = np.array(constraint[1:-1])
            nd = next(filter(lambda nd: nd.id == constraint[0], self.nodes), None)
            if nd is None: raise Exception('No matching node id in frame for constraint')
            else:
                # Fill con array with ones if only a few dimensions are given
                if len(con) < 6: con = np.append(con, np.ones(6 - len(con)))
                nd.fixed_dof = con

    def add_materials(self, material_list: list):
        # Create materials with provided data; Add to 'materials' list
        material_names = [existing_mat.name for existing_mat in self.materials]
        for mat in material_list:
            if isinstance(mat, material) and not mat.name in material_names:
                self.materials.append(mat)
                material_names.append(mat.name)
            elif isinstance(mat, list) and not mat[0] in material_names:
                new_material = material(*tuple(mat))
                self.materials.append(new_material)
                material_names.append(new_material.name)

    def add_xsections(self, xsection_list: list):
        xsection_ids = [xsec.id for xsec in self.xsections]
        for xsec in xsection_list:
            if isinstance(xsec, xsection) and not xsec.id in xsection_ids:
                self.xsections.append(xsec)
                xsection_ids.append(xsec.id)
            elif isinstance(xsec, list) and not xsec[0] in xsection_ids:
                new_xsection = xsection(*tuple(xsec))
                self.xsections.append(new_xsection)
                xsection_ids.append(new_xsection.id)

    def add_elements(self, elements):
        for el in elements:
            # Get node ids of element to add
            if isinstance(el[0], node): node_a = el[0].id
            else: node_a = el[0]
            if isinstance(el[1], node): node_b = el[1].id
            else: node_b = el[1]

            # Check if element already exists between specified nodes - create and add if not
            duplicate_element = False
            for existing_el in self.elements:
                node_pair = [existing_el.node_a.id, existing_el.node_b.id]
                if node_a in node_pair and node_b in node_pair:
                    duplicate_element = True
                    break
            if not duplicate_element: self.elements.append(element(*tuple([self] + el)))

    def rotate(self, rotation_angle: float, axis: list | np.ndarray = [1, 0, 0]):
        '''
        A function to return a copy of the frame rotated about a specified axis.
        Arguments:
          - axis: list | np.array = [1, 0, 0] - The axis of rotation specified by:
              1. x,y,z coordinates of a vector starting at the origin
              2. a pair of nodes
        '''
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
        return self.transform_affine(transform_matrix)

    def mirror(self, mirror_plane_pts: np.ndarray):
        ########### Add code to construct transformation matrix 
        return self.transform_affine(transform_matrix)

    def transform_affine(self, transform_matrix: np.ndarray):
        new_frame = cp.deepcopy(self)
        for i, nd in enumerate(self.nodes):
            transformed_coords = affine_matrix @ np.concat(nd.coords, np.array([1]))
            new_frame.nodes[i].coords = transformed_coords[0:2]
        return new_frame
    

    def discretize(self, atom_distance: float, start_atom_id: int = 0) -> tuple:
        '''
        A function to discretize a frame given a specified atom spacing using the element.discretize() method.
        It returns data of beams, bonds joining beams to the nodes, and angles between beams joined at the nodes,
        all formatted for straightforward use with the Simulation class' functions.

        Arguments:
         - atom_distance: float - the maximum distance permitted between atoms
         - start_atom_id: int = 0 - the number of atoms already in the simulation (offsets the atom ids)

        Outputs:
         - A tuple of data for beam, bond, and angle creation (beams, bonds, angles)
           - beams: list (# beams x 3) - [# atoms, start_coords, end_coords]
             - # atoms: int
             - start_coords: np.array (1 x 3) - x,y,z coordinates of an end atom
             - end_coords: np.array (1 x 3) - x,y,z coordinates of other end atom
           - bonds: np.array (#beam connections x 2) - [node_atom, end_atom]
             - node_atom: int - ID of atom at a node
             - end_atom: int - ID of atom at end of beam (to join with a bond to the atom at the node)
           - angles: np.array (#beam connections x 3) - [node_atom_1, node_atom_0, end_atom]
             - node_atom_1: int - ID of atom attached to the atom at a node
             - node_atom_0: int - ID of atom at the node
             - end_atom: int - ID of atom at end of beam (to join with an angle to the atom at the node)
        '''
        # Discretize the frame's elements into atoms (removing duplicate atoms at nodes)
        self.atoms = []
        self.start_atom_id = start_atom_id
        self.atoms = [el.discretize(atom_distance) for el in self.elements]
        beams = [[len(el.atoms), el.atoms[0], el.atoms[-1]] for el in self.elements]

        atom_pairs = []
        angles = []
        for nd in self.nodes:
            # Get ID of atom connected to the node's atom (to define the angle)
            if nd.elements[0].node_a is nd: adjacent_atom_id = nd.atom.id + 1
            elif nd.elements[0].node_b is nd: adjacent_atom_id = nd.atom.id - 1
            # Add a bond and angle for each element connected to node
            for el in nd.elements[1:]:
                # Ensure the atom at the far end of the beam is not mistakenly chosen
                if el.node_a is nd: end_atom = el.atoms[0].id
                elif el.node_b is nd: end_atom = el.atoms[-1].id
                atom_pairs.append([nd.atom.id, end_atom])
                angles.append([adjacent_atom_id, nd.atom.id, end_atom])
        bonds = np.array(atom_pairs)
        angles = np.array(angles)

        return (beams, bonds, angles)

class node:
    '''
    An object representing an endpoint or connection point between elements.

    Attributes:
        parent_frame: frame - 
        id: int - A unique ID (within the frame)
        coords: np.array (1 x 6) - 
        fixed_dof: np.array (1 x 6) = [0,0,0,0,0,0] - 
        atom: atom - 

    Methods:
        __init__ - 

        assign_connected_elements() - 
    '''
    def __init__(self, parent_frame: frame, coords: np.ndarray, constraint: np.ndarray = [0,0,0,0,0,0]):
        self.parent_frame = parent_frame
        self.id: int = len(parent_frame.nodes)
        self.coords = coords
        self.fixed_dof = constraint

    def assign_connected_elements(self):
        '''
        A function to list all elements connected to the node.
        This is important for discretization in allowing for the removal of coincident atoms at connected beam ends.
        '''
        self.elements = [el for el in self.parent_frame.elements if self in [el.node_a, el.node_b]]

    def edit_coords(self, new_coords: np.array):
        if len(new_coords) < 6: new_coords = np.append(new_coords, np.zeros(6 - len(new_coords)))
        #### TODO - Check connected elements to verify parametric eqs are still satisfied; Recalculate element lengths
        self.coords = new_coords

class xsection:
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

class material:
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
        return f'Name {self.name} - (E, v)=({self.E}, {self.v})'

class element:
    '''
    An object representing a parametric member extending between two nodes.

    Attributes:
        parent_frame: frame - The frame containing the element
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

        calc_arc_length() - Calculate arc length of a parametric curve in 3D

        set_material - Assign new or existing material to element

        set_xsection - Assign new or existing xsection to element

        discretize() - Discretize element into atoms separated by no more than a provided distance
    '''
    def __init__(
        self,
        parent_frame: frame,
        node_a: int | node,
        node_b: int | node,
        mat: list | material | str | None = None,
        xsec: list | xsection | int | None = None,
        para_eq: list = [],
        z_vec = []
        ):
        self.parent_frame = parent_frame
        # Assign end nodes
        if isinstance(node_a, int): 
            matching_node_a = next(filter(lambda nd: nd.id == node_a, parent_frame.nodes), None)
            if matching_node_a is None: raise Exception('No matching node found for element end a')
            self.node_a = matching_node_a
        else: 
            self.node_a = node_a
        if isinstance(node_b, int): 
            matching_node_b = next(filter(lambda nd: nd.id == node_b, parent_frame.nodes), None)
            if matching_node_b is None: raise Exception('No matching node found for element end b')
            self.node_b = matching_node_b
        else: 
            self.node_b = node_b

        # Set parametric equations for element path (if provided) and calculate element length
        self.set_parametric_eqs(para_eq)
        if len(para_eq) == 0: self.L = np.linalg.norm(self.node_b.coords - self.node_a.coords)
        else: self.L = self.calc_arc_length(para_eq)

        # Assign material and cross section - create instance if info is provided instead of objects
        # If not specified and the frame only has one, assign that one as the default
        if not mat is None: self.set_material(mat)
        elif len(self.parent_frame.materials) == 1: self.set_material(self.parent_frame.materials[0])
        if not xsec is None: self.set_xsection(xsec)
        elif len(self.parent_frame.xsections) == 1: self.set_xsection(self.parent_frame.xsections[0])

    def __str__(self):
        return f'Node Pair ({self.node_a.id}, {self.node_b.id}); X-Section {self.xsec.id}'

    def set_parametric_eqs(self, para_eq: list):
        # Handle parametric equation inputs - create linear parametric equations if none are provided
        if len(para_eq) > 0:
            para_eq.append([0 for i in range(3-len(para_eq))])

            # Replace any constants with callable functions
            for coord_eq in para_eq: 
                if not isinstance(coord_eq, callable): 
                    coord_eq = lambda t: coord_eq * t

            # Calculate node coordinates if they are not already set
            if len(self.node_a.coords) == 0:
                self.node_a.coords = np.array([para_eq[0](0), para_eq[1](0), para_eq[2](0)])
            if len(self.node_b.coords) == 0:
                self.node_b.coords = np.array([para_eq[0](1), para_eq[1](1), para_eq[2](1)])

            # Check all coordinates for mismatching values with parametric function
            for i_eq, coord_eq in enumerate(para_eq):
                if coord_eq(0) != self.node_a.coords[i_eq]: 
                    raise Exception(f'Parametric equation f(0) does not coincide with start node with ID {self.node_a.id}')
                if coord_eq(1) != self.node_b.coords[i_eq]: 
                    raise Exception(f'Parametric equation f(1) does not coincide with end node with ID {self.node_b.id}')
            self.para_eq = para_eq
        elif len(self.node_a.coords) > 0 and len(self.node_b.coords) > 0:
            x_eq = lambda t: (1-t)*self.node_a.coords[0] + t*self.node_b.coords[0]
            y_eq = lambda t: (1-t)*self.node_a.coords[1] + t*self.node_b.coords[1]
            z_eq = lambda t: (1-t)*self.node_a.coords[2] + t*self.node_b.coords[2]
            self.para_eq = [x_eq, y_eq, z_eq]

    def calc_arc_length(self, para_eq: list[callable], bounds: list = [0, 1], n_pts: int = 200) -> float:
        '''
        Numerically calculate the arc length of a curve defined by parametric equations between the provided bounds.
        It uses the numpy gradient and trapz functions, and allows the number of points (n_pts) to be specified.

        Arguments:
         - para_eq: list[callable] (1 x 3) - [x_eq(), y_eq(), z_eq()] defining x, y, and z with single parameter
         - bounds: list = [0, 1] (1 x 2) - the start and end values of the parameter to integrate between
         - n_pts: int = 200 - the number of points used to evaluate the parametric equations for numeric integration
        
        Outputs:
         - L: float - The calculated arc-length between the start and end points along the curve.
        '''
        t = np.linspace(bounds[0], bounds[1], n_pts)
        (x, y, z) = (para_eq[0](t), para_eq[1](t), para_eq[2](t))
        L = np.trapz(np.sqrt(np.gradient(x, t)**2 + np.gradient(y, t)**2 + np.gradient(z, t)**2), t)
        return L

    def set_material(self, mat: list | material | str):
        if isinstance(mat, list):
            self.parent_frame.add_materials([mat])
            self.material = next(filter(lambda mat_entry: mat_entry.name == mat[0], self.parent_frame.materials), None)
        elif isinstance(mat, material):
            self.parent_frame.add_materials([mat])
            self.material = next(filter(lambda mat_entry: mat_entry.name == mat.name, self.parent_frame.materials), None)
        else:
            mat = next(filter(lambda mat_entry: mat_entry.name == mat, parent_frame.materials), None)
            if mat is None: raise Exception('No material with matching name assigned to frame')
            else: self.material = mat

    def set_xsection(self, xsec: list | xsection | int):
        if isinstance(xsec, list):
            self.parent_frame.add_xsections([xsec])
            self.xsec = next(filter(lambda xsec_entry: xsec_entry.id == xsec[0], self.parent_frame.xsections), None)
        elif isinstance(xsec, xsection):
            self.parent_frame.add_xsections([xsec])
            self.xsec = next(filter(lambda xsec_entry: xsec_entry.id == xsec.id, self.parent_frame.xsections), None)
        else:
            frame_xsec = next(filter(lambda xsec_entry: xsec_entry.id == xsec, self.parent_frame.xsections), None)
            if frame_xsec is None: raise Exception('No xsection with matching id assigned to frame')
            else: self.xsec = frame_xsec

    def discretize(self, atom_distance: float) -> list:
        '''
        A function to discretize an element with spacing not greater than the specified atom distance.
        It removes overlapping atoms at nodes where elements join, and returns the list of atoms.
        '''
        ### Do not call except from frame.discretize() -- atom ids need to be fully reassigned
        # Compute atom coordinates with parametric equations
        N_atoms = 1 + int(np.ceil(self.L/atom_distance))
        t = np.linspace(0, 1, N_atoms)
        (x_eq, y_eq, z_eq) = tuple(self.para_eq)
        atom_coords = np.array([x_eq(t), y_eq(t), z_eq(t)]).T

        # Remove end atom coordinates if the created atoms would overlap with already existing atoms
        if not self.node_b.elements[0] is self: atom_coords = atom_coords[0:-2]
        if not self.node_a.elements[0] is self: atom_coords = atom_coords[1:-1]

        # Create the element's atoms; add to 'atoms' list; if end atoms, assign to coincident node
        self.atoms = [atom(coords, parent_element=self) for coords in atom_coords]
        if self.node_b.elements[0] is self: self.node_b.atom = self.atoms[-1]
        if self.node_a.elements[0] is self: self.node_a.atom = self.atoms[0]
        return self.atoms

class atom:
    def __init__(self, coords: np.array, *, N_previous_atoms: int = 0, parent_element: element | None = None):
        self.coords = coords
        if not parent_element is None:
            self.parent_element = parent_element
            parent_frame = parent_element.parent_frame
            N_previous_atoms = parent_frame.start_atom_id + len(parent_frame.atoms)
        self.id = N_previous_atoms

if __name__ == "__main__":
    nodes = [[0,0], [0,1], [1,1], [1,0]]
    (E, rho) = (0.96 * 10**6, 0.5)
    materials = [['test_material_0', E, rho]]
    beam_thickness = 0.002
    xsecs = [[0, beam_thickness]]
    elements = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
    constraints = [[0, 1,1,1,1,1,1], [1, 1,1,1,1,1,1]]
    new_frame = frame(nodes, material_list = materials, xsection_list = xsecs, element_list = elements, constraint_list = constraints)
    new_frame.discretize(0.02)
    print(new_frame)

class Simulation:

    def __init__(
        self,
        simulation_name: str,
        dimension: int,
        length: float,
        width: float,
        height: float,
        x_bound: str = "f",
        y_bound: str = "f",
        z_bound: str = "f",
    ):
        """
        Here we are going to initialize a bunch of stuff about the simulation.
        Most of ths is random LAMMPS stuff that needs to be in order so that
        we can make an elastogranular sim.
        Inputs:
        - simulation_name: the name of the folder that this script will create and write files to
        - dimension: 2 or 3 for a 2d or 3d simulation
        - length, width, height: the length, width, and height of the box that we are going to run the simulation in
        - (x,y,z)_bound: whether the x, y, or z boundaries are fixed (f) periodic (p) or shrink-wrapped (s)
        """
        # TODO add 2d

        print(
            "\n### Making a simulation file in the folder '"
            + simulation_name
            + "' ###\n"
        )

        self._dimension = dimension
        self._length = length
        self._width = width
        self._height = height
        self._x_bound = x_bound
        self._y_bound = y_bound
        self._z_bound = z_bound
        self._simulation_name = simulation_name
        self._particles = pd.DataFrame(
            columns=[
                "type",
                "x_position",
                "y_position",
                "z_position",
                "diameter",
                "density",
            ]
        )
        self._num_sets_of_grains = 0
        self._num_beams = 0
        self._num_loops = 0
        self._num_sheets = 0
        self._type_iter = 0
        self._bond_type_iter = 0
        self._angle_type_iter = 0
        self._connections_iter = 0
        self._perturb_iter = 0
        self._visc_iter = 0
        self._group_iter = 0
        self._move_iter = 0
        self._wall_iter = 0
        self._walls = []
        self._timestep = False
        self._have_run = False
        self._dump_file_every_this_many_seconds = False

        # Make the directory that the lammps simulation will be placed in
        self._path = os.path.join(os.getcwd(), self._simulation_name)
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        with open(os.path.join(self._path, "in.main_file"), "w") as f:
            # Atom style: angle, for bending and stretching rigidity, and sphere, for granular interactions
            f.write("atom_style hybrid angle sphere\n")
            # You can make this whatever you prefer! Consider lj for maximum eyebrow-raising potential
            f.write("units si\n")
            # We're just using one processor today
            f.write("processors	* * 1\n")
            # No idea what this does
            f.write("comm_modify vel yes\n")
            if dimension == 2:
                z_bound = "p"
                f.write("dimension 2")
            f.write("\n")
            # Define the box that the simulation is gonna happen in. First we create a region
            f.write(
                f"region simulationStation block -{length/2} {length/2} -{width/2} {width/2} -{height/2} {height/2}\n"
            )
            # Then we make it the simulation station. Here the numbers are:
            #   The number of types of atoms (100)
            #   The number of types of bonds (100000)
            #   The number of possible bonds per particle (100)
            #   The number of types of angles (10000)
            #   The number of possible angles per particle (100)
            # In this line we have to put something, and if we put too few, lammps will get mad at us. For example if we say we
            # are going to have 30 types of particles, but we insert 31 types if particles, LAMMPS will be like "thats more than we thought!"
            # and your sim will die. However, since we don't know how many we need at the time of the writing of this line,
            # We just put more that we think we'll need. And since we have lines like "bond_coeff * 0 0", we set all of the 
            # extra ones to zero. If you need to put in more than I allow here, feel free to change this line!
            f.write(
                "create_box 100 simulationStation bond/types 100000 extra/bond/per/atom 100 angle/types 10000 extra/angle/per/atom 100\n"
            )
            # Make the boundry fixed, periodic, or shrink wrapped
            f.write(
                "change_box	all boundary "
                + x_bound
                + " "
                + y_bound
                + " "
                + z_bound
                + "\n"
            )
            f.write("\n")
            # We will use a granular / lj pair style, giving us the ability to have granular interactions and also cohesive/repulsive energies
            f.write("pair_style hybrid/overlay granular lj/cut 0\n")
            # Set all of them to zero to start (LAMMPS doesnt like when anything is not set)
            f.write(
                "pair_coeff  * * granular hertz/material 0 0 0 tangential linear_nohistory 0 0\n"
            )
            f.write("pair_coeff  * * lj/cut 0 0 0\n")
            f.write("\n")
            # Not totally sure what this does
            f.write("special_bonds lj/coul 0 1.0 1.0\n")
            # These are the potentials we will use
            f.write("bond_style harmonic\n")
            f.write("angle_style cosine\n")
            f.write("\n")
            f.write("bond_coeff * 0 0\n")
            f.write("angle_coeff * 0\n")
            # Turn on integration
            f.write("fix integration all nve/sphere\n")
            # This allows you to put in more particles and bonds. You might need to change this if you want an even bigger simulation
            f.write("neigh_modify page 500000 one 50000\n")
            f.write("\n\n### Here we will begin to include particles ###\n\n")
        pass

    def add_grains(
        self, coords: np.array, diameter: float, density: float, filename: str = None
    ) -> int:
        """
        Add grains to the simulation
        Inputs:
        - coords: an Nx3 numpy array where N is the number of grains that we are trying to insert and
        the three columns are the x, y, and z coordinates of those grains
        - diameter: the diameter of the grains that you will be inserting
        - density: the density of the grains that you will be inserting
        - filename: the name of the file that you want to put the create lines in. I'd leave this empty
        Outputs:
        - The type id of the particles that you are adding. This is for input into methods like "add_pair_potential" or "remove_something"
        """

        self._type_iter += 1
        # Unless we are placing something specific, place grains
        if filename is None:
            print(f"\n### Adding {coords.shape[0]} grains to the simulation ###")
            self._num_sets_of_grains += 1
            filename = f"grains_{self._num_sets_of_grains}.txt"
        else:
            print(f"# Adding {coords.shape[0]} grains to the simulation #")
        # Add the "type" column to the coordinates as the first column of the array
        new_particles = np.vstack(
            (
                [self._type_iter] * coords.shape[0],
                coords.T,
                [diameter] * coords.shape[0],
                [density] * coords.shape[0],
            )
        ).T
        # Turn that into a dataframe
        if self._particles.shape[0]==0:
            indices = np.array(list(range(coords.shape[0])))+1
        else:
            indices = np.array(list(range(coords.shape[0])))+max(self._particles.index)+1
        new_particles = pd.DataFrame(new_particles, columns=self._particles.columns, index=indices)
        # Add those into the current set of particles
        self._particles = pd.concat([self._particles, new_particles], axis=0)
        # Put the particles into the simulation
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"\ninclude " + filename + " \n")
            f.write(f"set type {self._type_iter} diameter {diameter}\n")
            f.write(f"set type {self._type_iter} density {density}\n")
        with open(os.path.join(self._path, filename), "w") as f:
            for i in range(coords.shape[0]):
                f.write(
                    f"create_atoms {self._type_iter} single {coords[i][0]} {coords[i][1]} {coords[i][2]}\n"
                )

        # Change the timestep to be able to handle these grains
        if self._timestep:
            self._timestep = min(self._timestep, 0.1 * (diameter / 9.8) ** 0.5)
        else:
            self._timestep = 0.01 * (diameter / 9.8) ** 0.5

        return self._type_iter

    def add_beam(
        self,
        n_particles: int,
        pos1: np.array,
        pos2: np.array,
        geometric_thickness: float,
        youngs_modulus: float,
        density: float,
        energetic_thickness_multiplier: float = 1,
        strecthing_multiplier: float = 1,
        bending_multiplier: float = 1,
    ) -> Tuple[int, int, int]:
        """
        Create a beam out of particles, replete with bonds and angles to give the beam mechanical properties
        Inputs:
        - n_particles: The number of particles that you want to make up your beam
        - pos1: A 1x3 np.array with the x, y, z coordinates of one end of the beam
        - pos2: A 1x3 np.array with the x, y, z coordinates of the other end of the beam
        - geometric_thickness: The diameter of the particles which make up the beam
        - youngs_modulus: The Young's modulus of the beam
        - density: The density of the particles which make up the beam. NOTE that this is different from
            the density of the beam itself
        - energetic_thickness_multiplier: This is to change the thickness term in the stretching and bending
            energies, which changes the stretching energy linearly and the bending energy cubically
        - stretching_multiplier: In case you want to manually edit the stretching energy
        - bending_multiplier: In case you want to manually edit the bending energy
        Outputs:
        - The id of the particles, bonds, and angles that make up this beam
        """
        if n_particles<3:
            raise Exception("Your beam must have more than 3 particles")

        print(f"\n### Creating a beam with {n_particles} particles ###")

        self._num_beams += 1
        # Make a list of the coordinates of the new particles
        coords = pos1
        for i in range(1, n_particles):
            coords = np.vstack((coords, pos1 + (pos2 - pos1) * i / (n_particles - 1)))
        # Make a list of the particles that we want to connect via bonds
        n_so_far = max(self._particles.index) if self._particles.index.size > 0 else 0
        tuplets = [[n_so_far + 1, n_so_far + 2]]
        for i in range(2, n_particles):
            tuplets.append([n_so_far + i, n_so_far + i + 1])
        tuplets = np.array(tuplets)
        # Make a list of the particles that we want to connect via angles
        triplets = [[n_so_far + 1, n_so_far + 2, n_so_far + 3]]
        for i in range(2, n_particles - 1):
            triplets.append([n_so_far + i, n_so_far + i + 1, n_so_far + i + 2])
        triplets = np.array(triplets)

        self.add_grains(
            coords, geometric_thickness, density, f"beam_{self._num_beams}.txt"
        )

        # Thickness
        h = geometric_thickness * energetic_thickness_multiplier
        # rest length of the bonds is the spacing between the particles
        rest_length = np.linalg.norm(pos2 - pos1) / (n_particles - 1)
        # dfactor is the diameter of a particle divided by the distance between particles in the beam
        dfactor = geometric_thickness / rest_length
        bond_stiffness = strecthing_multiplier * youngs_modulus * h * dfactor / 2
        angle_stiffness = bending_multiplier * youngs_modulus * (h ** 3) * dfactor / 12

        self.construct_many_bonds(tuplets, bond_stiffness, rest_length)
        self.construct_many_angles(triplets, angle_stiffness)

        if youngs_modulus > 0:
            # Change the timestep to be able to handle these grains
            m = (4 / 3) * np.pi * (geometric_thickness * 0.5) ** 3
            k = bond_stiffness * 2  # Somehow include angle_stiffness
            if self._timestep:
                self._timestep = min(self._timestep, ((1 / 2) * m / k) ** 0.5)  # TODO
            else:
                self._timestep = ((1 / 2) * m / k) ** 0.5

        return self._type_iter, self._bond_type_iter, self._angle_type_iter

    def add_loop(
        self,
        n_particles: int,
        center: np.array,
        radius: float,
        normal: np.array,
        geometric_thickness: float,
        youngs_modulus: float,
        density: float,
        energetic_thickness_multiplier: float = 1,
        strecthing_multiplier: float = 1,
        bending_multiplier: float = 1,
    ) -> Tuple[int, int, int]:
        """
        Create a loop out of particles, replete with bonds and angles to give the loop mechanical properties
        Inputs:
        - n_particles: The number of particles that you want to make up your beam
        - center: A 1x3 np.array with the x, y, z coordinates of the center of the loop
        - radius: The radius of the loop
        - normal: A 1x3 np.array which gives the vector normal to the plane which the loop occupies,
            for example, a ring which is laying flat on a table would have a normal \porm [0, 0, 1]
        - geometric_thickness: The diameter of the particles which make up the beam
        - youngs_modulus: The Young's modulus of the beam
        - density: The density of the particles which make up the beam. NOTE that this is different from
            the density of the beam itself
        - energetic_thickness_multiplier: This is to change the thickness term in the stretching and bending
            energies, which changes the stretching energy linearly and the bending energy cubically
        - stretching_multiplier: In case you want to manually edit the stretching energy
        - bending_multiplier: In case you want to manually edit the bending energy
        Outputs:
        - The ids of the particles, bonds, and angles that make up this loop
        """
        self._num_beams += 1

        print(f"\n### Creating a loop with {n_particles} particles ###")

        # First I need to find two vectors that are perpendicular to the norm vector
        x1 = np.array([0, 0, 0])
        for i in range(3):
            if normal[i] != 0:
                x1[i - 1] = normal[i]
                x1[i] = -normal[i - 1]
                break
        x2 = np.cross(x1, normal)

        for vect in [normal, x1, x2]:
            vect = vect / np.linalg.norm(vect)

        # Now we start laying the coords
        coords = center + radius * x1
        thetas = np.linspace(2 * np.pi / n_particles, 2 * np.pi * (1 - 1 / n_particles), n_particles - 1)
        for theta in thetas:
            coords = np.vstack(
                (coords, center + radius * (x1 * np.cos(theta) + x2 * np.sin(theta)))
            )

        # Make a list of the particles that we want to connect via bonds
        n_so_far = max(self._particles.index) if self._particles.index.size > 0 else 0
        tuplets = [[n_so_far + n_particles, n_so_far + 1]]
        for i in range(1, n_particles):
            tuplets.append([n_so_far + i, n_so_far + i + 1])
        tuplets = np.array(tuplets)

        # Make a list of the particles that we want to connect via angles
        triplets = [[n_so_far + n_particles - 1, n_so_far + n_particles, n_so_far + 1]]
        triplets.append([n_so_far + n_particles, n_so_far + 1, n_so_far + 2])
        for i in range(1, n_particles - 1):
            triplets.append([n_so_far + i, n_so_far + i + 1, n_so_far + i + 2])
        triplets = np.array(triplets)

        self.add_grains(
            coords, geometric_thickness, density, f"beam_{self._num_beams}.txt"
        )

        # Thickness
        h = geometric_thickness * energetic_thickness_multiplier
        # rest length of the bonds is the spacing between the particles
        rest_length = np.linalg.norm(coords[0, :] - coords[1, :])
        # dfactor is the diameter of a particle divided by the distance between particles in the beam
        dfactor = geometric_thickness / rest_length
        bond_stiffness = strecthing_multiplier * youngs_modulus * h * dfactor / 2
        angle_stiffness = bending_multiplier * youngs_modulus * (h ** 3) * dfactor / 12
        self.construct_many_bonds(tuplets, bond_stiffness, rest_length)
        self.construct_many_angles(triplets, angle_stiffness)

        # Change the timestep to be able to handle these grains
        if youngs_modulus > 0:
            m = (4 / 3) * np.pi * (geometric_thickness * 0.5) ** 3
            k = bond_stiffness * 2  # Somehow include angle_stiffness
            if self._timestep:
                self._timestep = min(self._timestep, ((1 / 2) * m / k) ** 0.5)  # TODO
            else:
                self._timestep = ((1 / 2) * m / k) ** 0.5

        return self._type_iter, self._bond_type_iter, self._angle_type_iter

    def add_circular_sheet(
        self,
        center: np.array,
        radius: float,
        # normal: np.array,
        mesh_particle_spacing: float,
        energetic_thickness: float,
        youngs_modulus: float,
        density: float,
        mesh_particle_diameter: float = None,
        strecthing_multiplier: float = 1,
        bending_multiplier: float = 1,
    ) -> Tuple[int, int, int]:
        """
        Create a circular sheet out of particles, replete with bonds and angles to give the sheet mechanical properties
        NOTE: For now, all sheets have a normal of [0,0,1]. I might change this later (TODO)
        Inputs:
        - center: A 1x3 np.array with the x, y, z coordinates of the center of the sheet
        - radius: The radius of the sheet
        - mesh_particle_spacing: The resting distance between mesh particles in the sheet
        - energetic_thickness: This term is what is used to calculate the bending and stretching modulus of the sheet (along with the youngs modulus). 
            NOTE: changing this does not change any physical sizes in the simulation, only the energy in the bonds and angles between particles.
            Explicitly, for you elasticity mathematicians out there, this is h.
        - youngs_modulus: The Young's modulus of the sheet material
        - density: The density of the particles which make up the sheet. NOTE that this is different from
            the density of the sheet itself
        - mesh_particle_diameter: This term changes the diameter of the mesh particles. If this is not set (left set to None),
            the mesh particle diameter will simply be set to the mesh particle spacing, such that all of the particles in the sheet
            will look like they're right next to each other
        - stretching_multiplier: In case you want to manually edit the stretching energy
        - bending_multiplier: In case you want to manually edit the bending energy
        Outputs:
        - The ids of the particles, bonds, and angles that make up this sheet
        """
        self._num_sheets += 1

        geometric_thickness = mesh_particle_diameter if mesh_particle_diameter else mesh_particle_spacing

        coords = [[0, 0, 0]]
        for i in range(
            round(-radius * 2 / mesh_particle_spacing), round(radius * 2 / mesh_particle_spacing)
        ):
            for j in range(
                round(-radius * 2 / mesh_particle_spacing),
                round(radius * 2 / mesh_particle_spacing),
            ):
                if i % 2 == 0:
                    x = i * np.cos(np.pi / 6) * mesh_particle_spacing
                    y = j * mesh_particle_spacing
                else:
                    x = i * np.cos(np.pi / 6) * mesh_particle_spacing
                    y = (j - np.sin(np.pi / 6)) * mesh_particle_spacing
                if (x ** 2 + y ** 2) ** 0.5 < radius:
                    coords.append(list(center + np.array([x, y, 0])))
        del coords[0]
        coords = np.array(coords)

        print(f"\n### Creating a circular sheet with {coords.shape[0]} particles ###")

        n_so_far = max(self._particles.index) if self._particles.index.size > 0 else 0
        tuplets = [[0, 0]]
        for i in range(coords.shape[0]):
            a = 0
            for j in range(i + 1, coords.shape[0]):
                if _dist(coords,i,j)< mesh_particle_spacing * 1.01:
                    tuplets.append([n_so_far + i + 1, n_so_far + j + 1])
                    a += 1
                    if a == 3:break
        del tuplets[0]
        tuplets = np.array(tuplets)

        triplets = [[0, 0, 0]]
        for i in range(coords.shape[0]):
            a = 0
            for j in range(i + 1, coords.shape[0]):
                if _dist(coords,i,j) < mesh_particle_spacing * 1.01:
                    for k in range(j + 1, coords.shape[0]):
                        if (
                            _dist(coords,j,k) < mesh_particle_spacing * 1.01
                            and _dist(coords,i,k) > mesh_particle_spacing * 1.9
                        ):
                            triplets.append([n_so_far + i + 1, n_so_far + j + 1, n_so_far + k + 1])
                            break
                    a += 1
                    if a == 3:break

        del triplets[0]
        triplets = np.array(triplets)

        self.add_grains(
            coords, geometric_thickness, density, f"sheet_{self._num_sheets}.txt"
        )
        # Thickness
        h = energetic_thickness
        # rest length of the bonds is the spacing between the particles
        rest_length = mesh_particle_spacing
        # dfactor is the diameter of a particle divided by the distance between particles in the beam
        bond_stiffness = (
            strecthing_multiplier * youngs_modulus * h * (3 ** 0.5) / 4
        )
        angle_stiffness = (
            (4 / (3 * (3 ** 2)))
            * bending_multiplier
            * youngs_modulus
            * (h ** 3)
            / (12 * (1 - (1 / 3) ** 2))
        )

        self.construct_many_bonds(tuplets, bond_stiffness, rest_length)
        self.construct_many_angles(triplets, angle_stiffness)

        if youngs_modulus > 0:
            m = (4 / 3) * np.pi * (geometric_thickness * 0.5) ** 3
            k = bond_stiffness * 2  # Somehow include angle_stiffness
            if self._timestep:
                self._timestep = min(self._timestep, ((1 / 6) * m / k) ** 0.5)  # TODO
            else:
                self._timestep = ((1 / 6) * m / k) ** 0.5

        return self._type_iter, self._bond_type_iter, self._angle_type_iter

    def add_rectangular_sheet(
        self,
        center: np.array,
        side_length1: float,
        side_length2: float,
        # normal: np.array,
        mesh_particle_spacing: float,
        energetic_thickness: float,
        youngs_modulus: float,
        density: float,
        mesh_particle_diameter: float = None,
        strecthing_multiplier: float = 1,
        bending_multiplier: float = 1,
    ) -> Tuple[int, int, int]:
        """
        Create a rectangular sheet out of particles, replete with bonds and angles to give the sheet mechanical properties
        NOTE: For now, all sheets have a normal of [0,0,1]. I might change this later (TODO)
        Inputs:
        - center: A 1x3 np.array with the x, y, z coordinates of the center of the sheet
        - side_length1: The x-dimension length of the sheet
        - side_length2: The y-dimension length of the sheet
        - mesh_particle_spacing: The resting distance between mesh particles in the sheet
        - energetic_thickness: This term is what is used to calculate the bending and stretching modulus of the sheet (along with the youngs modulus). 
            NOTE: changing this does not change any physical sizes in the simulation, only the energy in the bonds and angles between particles.
            Explicitly, for you elasticity mathematicians out there, this is h.
        - youngs_modulus: The Young's modulus of the sheet material
        - density: The density of the particles which make up the sheet. NOTE that this is different from
            the density of the sheet itself
        - mesh_particle_diameter: This term changes the diameter of the mesh particles. If this is not set (left set to None),
            the mesh particle diameter will simply be set to the mesh particle spacing, such that all of the particles in the sheet
            will look like they're right next to each other
        - stretching_multiplier: In case you want to manually edit the stretching energy
        - bending_multiplier: In case you want to manually edit the bending energy
        Outputs:
        - The ids of the particles, bonds, and angles that make up this sheet
        """
        self._num_sheets += 1

        geometric_thickness = mesh_particle_diameter if mesh_particle_diameter else mesh_particle_spacing

        coords = [[0, 0, 0]]
        for i in range(
            round(- side_length1 * 1.5 / mesh_particle_spacing), round(side_length1 * 1.5 / mesh_particle_spacing)
        ):
            for j in range(
                round(-side_length2 * 1.5 / mesh_particle_spacing),
                round(side_length2 * 1.5 / mesh_particle_spacing),
            ):
                if i % 2 == 0:
                    x = i * np.cos(np.pi / 6) * mesh_particle_spacing
                    y = j * mesh_particle_spacing
                else:
                    x = i * np.cos(np.pi / 6) * mesh_particle_spacing
                    y = (j - np.sin(np.pi / 6)) * mesh_particle_spacing
                if abs(x) < side_length1 / 2 and abs(y) < side_length2 / 2 :
                    coords.append(list(center + np.array([x, y, 0])))
        del coords[0]
        coords = np.array(coords)

        print(f"\n### Creating a circular sheet with {coords.shape[0]} particles ###")

        n_so_far = max(self._particles.index) if self._particles.index.size > 0 else 0
        tuplets = [[0, 0]]
        for i in range(coords.shape[0]):
            a = 0
            for j in range(i + 1, coords.shape[0]):
                if _dist(coords,i,j)< mesh_particle_spacing * 1.01:
                    tuplets.append([n_so_far + i + 1, n_so_far + j + 1])
                    a += 1
                    if a == 3:break
        del tuplets[0]
        tuplets = np.array(tuplets)

        triplets = [[0, 0, 0]]
        for i in range(coords.shape[0]):
            a = 0
            for j in range(i + 1, coords.shape[0]):
                if _dist(coords,i,j) < mesh_particle_spacing * 1.01:
                    for k in range(j + 1, coords.shape[0]):
                        if (
                            _dist(coords,j,k) < mesh_particle_spacing * 1.01
                            and _dist(coords,i,k) > mesh_particle_spacing * 1.9
                        ):
                            triplets.append([n_so_far + i + 1, n_so_far + j + 1, n_so_far + k + 1])
                            break
                    a += 1
                    if a == 3:break

        del triplets[0]
        triplets = np.array(triplets)

        self.add_grains(
            coords, geometric_thickness, density, f"sheet_{self._num_sheets}.txt"
        )
        # Thickness
        h = energetic_thickness
        # rest length of the bonds is the spacing between the particles
        rest_length = mesh_particle_spacing
        # dfactor is the diameter of a particle divided by the distance between particles in the beam
        bond_stiffness = (
            strecthing_multiplier * youngs_modulus * h * (3 ** 0.5) / 4
        )
        angle_stiffness = (
            (4 / (3 * (3 ** 2)))
            * bending_multiplier
            * youngs_modulus
            * (h ** 3)
            / (12 * (1 - (1 / 3) ** 2))
        )

        self.construct_many_bonds(tuplets, bond_stiffness, rest_length)
        self.construct_many_angles(triplets, angle_stiffness)

        if youngs_modulus > 0:
            m = (4 / 3) * np.pi * (geometric_thickness * 0.5) ** 3
            k = bond_stiffness * 2  # Somehow include angle_stiffness
            if self._timestep:
                self._timestep = min(self._timestep, ((1 / 6) * m / k) ** 0.5)  # TODO
            else:
                self._timestep = ((1 / 6) * m / k) ** 0.5

        return self._type_iter, self._bond_type_iter, self._angle_type_iter

    def add_cylindrical_sheet(
        self,
        bottom_center: np.array,
        radius: float,
        height: float,
        normal: np.array,
        mesh_particle_spacing: float,
        energetic_thickness: float,
        youngs_modulus: float,
        density: float,
        mesh_particle_diameter: float = None,
        strecthing_multiplier: float = 1,
        bending_multiplier: float = 1,
    ) -> Tuple[int, int, int]:
        """
        Create a cylindrical sheet out of particles, replete with bonds and angles to give the sheet mechanical properties
        Inputs:
        - center: A 1x3 np.array with the x, y, z coordinates of the center of the bottom ring of the cylinder
        - radius: The radius of the sheet
        - normal: The vector which points down the center of the cylinder
        - mesh_particle_spacing: The resting distance between mesh particles in the sheet
        - energetic_thickness: This term is what is used to calculate the bending and stretching modulus of the sheet (along with the youngs modulus). 
            NOTE: changing this does not change any physical sizes in the simulation, only the energy in the bonds and angles between particles.
            Explicitly, for you elasticity mathematicians out there, this is h.
        - youngs_modulus: The Young's modulus of the sheet material
        - density: The density of the particles which make up the sheet. NOTE that this is different from
            the density of the sheet itself
        - mesh_particle_diameter: This term changes the diameter of the mesh particles. If this is not set (left set to None),
            the mesh particle diameter will simply be set to the mesh particle spacing, such that all of the particles in the sheet
            will look like they're right next to each other
        - stretching_multiplier: In case you want to manually edit the stretching energy
        - bending_multiplier: In case you want to manually edit the bending energy
        Outputs:
        - The ids of the particles, bonds, and angles that make up this sheet
        """
        self._num_sheets += 1

        geometric_thickness = mesh_particle_diameter if mesh_particle_diameter else mesh_particle_spacing

        # First I need to find two vectors that are perpendicular to the norm vector
        x1 = np.array([0, 0, 0])
        for i in range(3):
            if normal[i] != 0:
                x1[i - 1] = normal[i]
                x1[i] = -normal[i - 1]
                break
        x2 = np.cross(x1, normal)

        for vect in [normal, x1, x2]:
            vect = vect / np.linalg.norm(vect)

        dth = 2 * np.pi / np.round(2 * np.pi * radius / (mesh_particle_spacing * 3 ** 0.5))
        thetas1 = np.arange(0,2 * np.pi,dth)
        thetas2 = thetas1 + dth / 2
        x = radius * np.cos(thetas1)
        y = radius * np.sin(thetas1)
        actuald = (((x[1]-x[0]) ** 2 + (y[1]-y[0]) **2) ** 0.5)/(3 ** 0.5)
        print(actuald)

        numrings = round(2*height/actuald)
        coords = [[0,0,0]]
        for ring in range(numrings):
            thetas = thetas1 if ring % 2 == 0 else thetas2
            for theta in thetas:
                coords.append(list(bottom_center + radius * (x1 * np.cos(theta) + x2 * np.sin(theta)) + normal * ring * actuald / 2))
        del coords[0]
        coords = np.array(coords)

        print(f"\n### Creating a circular sheet with {coords.shape[0]} particles ###")

        mesh_particle_spacing = actuald

        n_so_far = max(self._particles.index) if self._particles.index.size > 0 else 0
        tuplets = [[0, 0]]
        for i in range(coords.shape[0]):
            a = 0
            for j in range(i + 1, coords.shape[0]):
                if _dist(coords,i,j)< mesh_particle_spacing * 1.01:
                    tuplets.append([n_so_far + i + 1, n_so_far + j + 1])
                    a += 1
                    if a == 3:break
        del tuplets[0]
        tuplets = np.array(tuplets)

        triplets = [[0, 0, 0]]
        for i in range(coords.shape[0]):
            a = 0
            for j in range(i + 1, coords.shape[0]):
                if _dist(coords,i,j) < mesh_particle_spacing * 1.05:
                    for k in range(j + 1, coords.shape[0]):
                        if (
                            _dist(coords,j,k) < mesh_particle_spacing * 1.05
                            and _dist(coords,i,k) > mesh_particle_spacing * 1.9
                        ):
                            triplets.append([n_so_far + i + 1, n_so_far + j + 1, n_so_far + k + 1])
                            break
                    a += 1
                    if a == 3:break

        del triplets[0]
        triplets = np.array(triplets)

        self.add_grains(
            coords, geometric_thickness, density, f"sheet_{self._num_sheets}.txt"
        )
        # Thickness
        h = energetic_thickness
        # rest length of the bonds is the spacing between the particles
        rest_length = mesh_particle_spacing
        # dfactor is the diameter of a particle divided by the distance between particles in the beam
        bond_stiffness = (
            strecthing_multiplier * youngs_modulus * h * (3 ** 0.5) / 4
        )
        angle_stiffness = (
            (4 / (3 * (3 ** 2)))
            * bending_multiplier
            * youngs_modulus
            * (h ** 3)
            / (12 * (1 - (1 / 3) ** 2))
        )

        self.construct_many_bonds(tuplets, bond_stiffness, rest_length)
        self.construct_many_angles(triplets, angle_stiffness)

        if youngs_modulus > 0:
            m = (4 / 3) * np.pi * (geometric_thickness * 0.5) ** 3
            k = bond_stiffness * 2  # Somehow include angle_stiffness
            if self._timestep:
                self._timestep = min(self._timestep, ((1 / 6) * m / k) ** 0.5)  # TODO
            else:
                self._timestep = ((1 / 6) * m / k) ** 0.5

        return self._type_iter, self._bond_type_iter, self._angle_type_iter

    def construct_many_bonds(
        self, tuplets: np.array, stiffness: float, rest_length: float
    ) -> int:
        """
        Add harmonic bonds between particles.
        Inputs:
        - tuplets: An Nx2 np.array where the rows are pairs of particles that you want to make a bond between
        - stiffness: The stiffness of these bonds
        - rest_length: The rest length of the bonds
        Outputs:
        - The ids of these bonds
        """
        self._bond_type_iter += 1

        print(f"# Creating {tuplets.shape[0]} Bonds #")

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"bond_coeff {self._bond_type_iter} {stiffness} {rest_length}\n")
            f.write(f"include bonds_{self._bond_type_iter}.txt\n")
        with open(
            os.path.join(self._path, f"bonds_{self._bond_type_iter}.txt"), "w"
        ) as f:
            for i in range(tuplets.shape[0]):
                f.write(
                    f"create_bonds single/bond {self._bond_type_iter} {tuplets[i][0]} {tuplets[i][1]}\n"
                )

        return self._bond_type_iter

    def construct_many_angles(self, triplets: np.array, stiffness: float) -> int:
        """
        Add cosine angles between particles.
        Inputs:
        - triplets: An Nx3 np.array where the rows are triplets of particles that you want to make an angle between
        - stiffness: The stiffness of these angles
        Outputs:
        - The ids of these angles
        """
        self._angle_type_iter += 1

        print(f"# Creating {triplets.shape[0]} Angles #")

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"angle_coeff {self._angle_type_iter} {stiffness}\n")
            f.write(f"include angles_{self._angle_type_iter}.txt\n")
        with open(
            os.path.join(self._path, f"angles_{self._angle_type_iter}.txt"), "w"
        ) as f:
            for i in range(triplets.shape[0]):
                f.write(
                    f"create_bonds single/angle {self._angle_type_iter} {triplets[i][0]} {triplets[i][1]} {triplets[i][2]}\n"
                )

        return self._angle_type_iter

    def turn_on_granular_potential(
        self, type1: int = None, type2: int = None, youngs_modulus: float = None, hardcore_dict: dict = None
    ):
        # restitution = None, poissons = None, xscaling = None, coeffric = None
        """
        Make two types of particles interact in a granular way. This can either be a simple
        contact potential, which repels particles which are overlapping, or it can be a super
        complicated potential which adds all sorts of granular mechanics. If you want the chill
        potentiall, all you need to input is:
        - type1, type2: The types of particles you want to add an interaction potential to. This is output from methods like "add_grains," "add_beam," etc. 
        - youngs_modulus: The youngs modulus of the particles -- this will determine how much they can overlap given some confining force

        If you don't pass in type2, it will turn on the granular potential between type1 and all other particles. If you
        pass in neither type1 nor type2 it will turn on the granular potential between all particles.
        
        If you want to add additional physics, such that these particles actually
        behave like some sort of granular material, then you should:
        - input a "hardcore_dict" which contains:
            - restitution: The restitution coefficient
            - poissons: The Poisson's ratio of the interaction
            - xscaling: A scalar multiplier for the tangential damping
            - coeffric: The coefficient of sliding friction
            - gammar: The degree of rolling damping
            - rolfric: The coefficient of rolling friction
        I'd check out the lammps documentation for the "pair_style granular command" if you wanna be hardcore. Note that the downside
        to being hardcore is that it makes the simulation take much longer
        """

        if hardcore_dict and not all(
            key in hardcore_dict
            for key in (
                "restitution",
                "poissons",
                "xscaling",
                "coeffric",
                "gammar",
                "rolfric",
            )
        ):
            raise Exception(
                "If the hardcore flag is on, you also need to input 'restitution', 'poissons', 'xscaling', 'coeffric', 'gammar', 'rolfric' in the hardcore_dict"
            )

        if not type1 is None and not type2 is None:
            a = [type1, type2]
            type1 = min(a)
            type2 = max(a)

        print(
            f"\n### Initiating a granular potential between type {'all' if type1 is None else type1} and type {'all' if type2 is None else type2} Particles ###"
        )

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            if hardcore_dict:
                kr = (
                    4
                    * youngs_modulus
                    / (
                        2
                        * (2 - hardcore_dict["poissons"])
                        * (1.0 + hardcore_dict["poissons"])
                    )
                )
                f.write(f"pair_coeff {'*' if type1 is None else type1} {'*' if type2 is None else type2} granular &\n")
                f.write(
                    f"hertz/material {youngs_modulus} {hardcore_dict['restitution']} {hardcore_dict['poissons']} tangential mindlin NULL {hardcore_dict['xscaling']} {hardcore_dict['coeffric']} &\n"
                )
                f.write(
                    f"rolling sds {kr} {hardcore_dict['gammar']} {hardcore_dict['rolfric']} twisting marshall damping tsuji\n"
                )
            else:
                f.write(
                    f"pair_coeff {'*' if type1 is None else type1} {'*' if type2 is None else type2} granular hertz/material {youngs_modulus} 0 0.5 tangential linear_nohistory 0 0\n"
                )
        pass

    def turn_on_cohesive_potential(
        self, type1: int = None, type2: int = None, cohesive_strength: float = None, rest_length: float = None, cutoff:float = None
    ):
        """
        Make two types of particles have a cohesive (or repellant) lj potential. This takes in:
        - type1, type2: The types of particles you want to add an interaction potential to. This is output from methods like "add_grains," "add_beam," etc. 
        - cohesive_strength: This is epsilon in https://docs.lammps.org/pair_lj.html
        - rest_length: The rest length of the bonds. This is NOT sigma in https://docs.lammps.org/pair_lj.html
        - cutoff: If two particles get farther than "cutoff" apart, they will not be coherent any more. This can just be really big,
            or like 2.5 times rest_length, where the potential would pretty much disappear anyways

        If you don't pass in type2, it will turn on the potential between type1 and all other particles. If you
        pass in neither type1 nor type2 it will turn on the potential between all particles.
        
        """

        if any(thing is None for thing in [cohesive_strength, rest_length, cutoff]):
            raise Exception(
                "You must pass in a bond_strength, rest_length, and cutoff, sorry :/. The cutoff can just be really big if you don't actually want one"
            )

        if not type1 is None and not type2 is None:
            a = [type1, type2]
            type1 = min(a)
            type2 = max(a)

        print(
            f"\n### Initiating an lj potential between type {'all' if type1 is None else type1} and type {'all' if type2 is None else type2} Particles ###"
        )

        sigma = rest_length / ( 2 ** (1/6) )

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(
                f"pair_coeff {'*' if type1 is None else type1} {'*' if type2 is None else type2} lj/cut {cohesive_strength} {sigma} {cutoff}\n"
            )
        pass

    def connect_particles_to_elastic(
        self,
        elastic_type: int,
        stiffness: float,
        particles: List[int] = None,
        type: int = None,
        n_bonds_per_grain: int = 3,
        # cutoff: float = None,
    ) -> int:
        """
        TODO: Add cutoff

        Create bonds between a set of particles and an elastic material. This will set the rest length of the bonds equal to the
        distance, at the time of the command, between each particle and the surface.
        Inputs:
        EITHER
        - particles: a list of particles
        OR 
        - type: a type of particles
        and:
        - elastic_type: The type of the elastic
        - stiffness: The stiffness of the bonds
        - n_bonds_per_particle: How many connections you want between each particle and the elastic. This is to restrict degrees of freedom. This can be 1 2 or 3.

        Note that this cannot be undone once it is done, unless you delete the particles or the surface
        """

        if n_bonds_per_grain > 3:
            raise Exception(f"Can't do more than 3 bonds per grain, you tried to do {n_bonds_per_grain}")
        
        if not particles is None and not type is None:
            raise Exception("Can either pass in a list of particles, or a type of particles, not both")

        self._have_run = True
        self._connections_iter += 1

        ## If we get a type of particles, turn particles into a list of the particles of that type
        if not type is None:
            print(
                f"\n### Connecting particles of type {type} to the surface of type {elastic_type} ###"
            )
            particles = self._particles.index[
                self._particles["type"] == type
            ].tolist()
        else:
            print(
                f"\n### Connecting A list of particles to the surface of type {elastic_type} ###"
            )
        #print(particles)
        filename = f"connect_{self._connections_iter}.txt"
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"\ninclude " + filename + " \n")

        with open(os.path.join(self._path, filename), "w") as f:
            f.write(f"group elastic_connect type {elastic_type}")
            for particle in particles:
                f.write(f"\n\n\ngroup now id {particle}")
                f.write("\nvariable distances atom abs(sqrt((xcm(now,x)-x)^2+(xcm(now,y)-y)^2+(xcm(now,z)-z)^2))")
                f.write("\nvariable rnow equal sqrt((xcm(now,x))^2+(xcm(now,y))^2)")
                f.write("\ncompute cmindist elastic_connect reduce min v_distances")
                f.write("\nthermo_style	custom step atoms time c_cmindist")
                f.write("\nrun 0")
                f.write("\nthermo_style custom step atoms time")
                f.write("\nvariable mindist equal c_cmindist")
                f.write("\nvariable rhinow equal ${mindist}*1.0000001")
                self._bond_type_iter += 1
                f.write(f"\nbond_coeff {self._bond_type_iter} {stiffness} " + "${mindist}")
                f.write(f"\ncreate_bonds many now elastic_connect {self._bond_type_iter} 0 "+"${rhinow}")
                if n_bonds_per_grain > 1:
                    f.write("\nvariable xone equal xcm(now,x)")
                    f.write("\nvariable yone equal xcm(now,y)")
                    f.write("\nvariable zone equal xcm(now,z)")
                    f.write("\nregion roneplus sphere ${xone} ${yone} ${zone} ${rhinow}")
                    f.write("\ngroup goneplus region roneplus")
                    f.write("\ngroup notTheClosest subtract all goneplus")
                    f.write("\nuncompute cmindist")
                    f.write("\ncompute cmindist notTheClosest reduce min v_distances")
                    f.write("\nthermo_style custom step atoms time c_cmindist")
                    f.write("\nrun 0")
                    f.write("\nthermo_style custom step atoms time")
                    f.write("\nvariable mindist equal c_cmindist")
                    f.write("\nvariable rhinow equal ${mindist}*1.0000001")
                    self._bond_type_iter += 1
                    f.write(f"\nbond_coeff {self._bond_type_iter} {stiffness} " + "${mindist}")
                    f.write(f"\ncreate_bonds many now notTheClosest {self._bond_type_iter} 0 "+"${rhinow}")
                    if n_bonds_per_grain > 2:
                        f.write("\nregion rtwoplus sphere ${xone} ${yone} ${zone} ${rhinow}")
                        f.write("\ngroup gtwoplus region rtwoplus")
                        f.write("\ngroup notTheClosestOrTheSecondClosest subtract all gtwoplus")
                        f.write("\nuncompute cmindist")
                        f.write("\ncompute cmindist notTheClosestOrTheSecondClosest reduce min v_distances")
                        f.write("\nthermo_style custom step atoms time c_cmindist")
                        f.write("\nrun 0")
                        f.write("\nthermo_style custom step atoms time")
                        f.write("\nvariable mindist equal c_cmindist")
                        f.write("\nvariable rhinow equal ${mindist}*1.0000001")
                        self._bond_type_iter += 1
                        f.write(f"\nbond_coeff {self._bond_type_iter} {stiffness} " + "${mindist}")
                        f.write(f"\ncreate_bonds many now notTheClosestOrTheSecondClosest {self._bond_type_iter} 0 "+"${rhinow}")
                f.write("\nuncompute cmindist")
                f.write("\ngroup now delete")
                if n_bonds_per_grain > 2:
                    f.write("\ngroup notTheClosestOrTheSecondClosest delete")
                    f.write("\ngroup gtwoplus delete")
                    f.write("\nregion rtwoplus delete")
                if n_bonds_per_grain > 1:
                    f.write("\ngroup notTheClosest delete")
                    f.write("\ngroup goneplus delete")
                    f.write("\nregion roneplus delete")
            f.write(f"\ngroup elastic_connect delete")
        return self._connections_iter

    def add_walls(
        self,
        region_details: str = None,
        particles: Union[List[int],int] = None,
        type: Union[List[int],int] = None,
        youngs_modulus: float = None,
        hardcore_dict: dict = None,
    ) -> int:
        """
        Make a lammps "region", and optionally make the walls of that region have a granular potential with a type of particles.
        This can either be a simple contact potential, which repels particles which are overlapping, or it can be a super
        complicated potential which adds all sorts of granular mechanics. To make the region, you have to add the
        - region_details: Everything that is required to define a lammps region other than the region name. 
        For example, if I wanted to define a cylinder which was aligned with the z axis and whose center was at x=0, y=0
        with a radius of 0.5 and which went from z=-0.2 to z=0.2 then I would have region_details = "cylinder z 0 0 0.5 -0.2 0.2".
        The full set of options are detailed here: https://docs.lammps.org/region.html
        
        If you pass in a youngs_modulus, the walls of the region that you are defining will become stiff to 
        some or all of the grains. Further options are:
        - If you pass in neither a particle or list of particle ids (particles) nor a type or list of types of particles,
            ALL particles in the simulation will interact with the walls
        - If you pass in either particles, or type, the particles or type of particles that you pass in will interact with the walls
        - If you want to add additional physics, such that these particles actually behave like some sort of granular material, then you should input a "hardcore_dict" which contains:
            - restitution: The restitution coefficient
            - poissons: The Poisson's ratio of the interaction
            - xscaling: A scalar multiplier for the tangential damping
            - coeffric: The coefficient of sliding friction
            - gammar: The degree of rolling damping
            - rolfric: The coefficient of rolling friction
        I'd check out the lammps documentation for the "pair_style granular command" if you wanna be hardcore. Note that the downside
        to being hardcore is that it makes the simulation take much longer

        One more thing -- if you pass in nothing for the region details, this will make the walls of the simulation station hard to the
        particles or type that you input, with youngs modulus and hardcore dict acting as usual. Note that this cannot currently be undone (TODO)
        """
        if hardcore_dict and not all(
            key in hardcore_dict
            for key in (
                "restitution",
                "poissons",
                "xscaling",
                "coeffric",
                "gammar",
                "rolfric",
            )
        ):
            raise Exception(
                "If the hardcore flag is on, you also need to input 'restitution', 'poissons', 'xscaling', 'coeffric', 'gammar', 'rolfric' in the hardcore_dict"
            )

        if not particles is None and not type is None:
            raise Exception(
                "You can make EITHER a particle OR a type of particle OR all of the particles in the simulation (input neither particles nor type) interact with a wall"
            )
        
        if not type is None:
            if isinstance(type,int):
                group = f"type {type}"
            else: 
                group = f"type " + " ".join([str(typ) for typ in type])
        elif not particles is None:
            if isinstance(particles,int):
                group = f"id {particles}"
            else: 
                group = f"id " + " ".join([str(part) for part in particles])
        else: group = "all"

        self._wall_iter += 1
        self._walls.append([bool(particles or type), bool(youngs_modulus)])

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write("\n")
            if group != "all":
                f.write(f"group group_wall_{self._wall_iter} " + group + "\n")
                group = f"group_wall_{self._wall_iter}"

            if not region_details is None:
                f.write(f"region region_{self._wall_iter} " + region_details + "\n")
                region = f"region_{self._wall_iter}"
            else:
                region = "simulationStation"

            if youngs_modulus:
                if hardcore_dict:
                    kr = (
                        4
                        * youngs_modulus
                        / (
                            2
                            * (2 - hardcore_dict["poissons"])
                            * (1.0 + hardcore_dict["poissons"])
                        )
                    )
                    f.write(
                        f"fix fix_walls_{self._wall_iter} "
                        + group
                        + " wall/gran/region granular &\n"
                    )
                    f.write(
                        f"hertz/material {youngs_modulus} {hardcore_dict['restitution']} {hardcore_dict['poissons']} tangential mindlin NULL {hardcore_dict['xscaling']} {hardcore_dict['coeffric']} &\n"
                    )
                    f.write(
                        f"rolling sds {kr} {hardcore_dict['gammar']} {hardcore_dict['rolfric']} twisting marshall damping tsuji region " + region + "\n"
                    )
                else:
                    f.write(
                        f"fix fix_walls_{self._wall_iter} "
                        + group
                        + f" wall/gran/region granular hertz/material {youngs_modulus} 0.25 0.25 tangential linear_nohistory 0 0 region " + region + "\n"
                    )
        return self._wall_iter

    def move(
        self,
        particles: Union[List[int], int] = None,
        type: Union[List[int],int] = None,
        xvel: float = None,
        yvel: float = None,
        zvel: float = None,
    ) -> int:
        """
        Move a set of particles at some velocity
        Pass in EITHER:
        - particles: A partice or list of particles to move
        OR
        - type: A type or list of types of particle to move (output from methods like "add_grains")

        And:
        - xvel, yvel, zvel: The velocity in the x, y, and z directions to move that particle

        Note: If you pass in 'None' to either xvel, yvel, or zvel, or leave them blank, those 
        velocities will not be mandated, and the particles will be free to move in those directions
        """
        if not particles is None and not type is None:
            raise Exception("You can move EITHER a particle OR a type of particle")
        if particles is None and type is None:
            raise Exception("You must move either a particle or a type of particle")
        
        if not type is None:
            if isinstance(type,int):
                group = f"type {type}"
            else: 
                group = f"type " + " ".join([str(typ) for typ in type])
        else:
            if isinstance(particles,int):
                group = f"id {particles}"
            else: 
                group = f"id " + " ".join([str(part) for part in particles])

        self._move_iter += 1

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"\ngroup group_move_{self._move_iter} " + group + "\n")
            f.write(
                f"fix fix_move_{self._move_iter} group_move_{self._move_iter} move linear "
                + " ".join(str(dir_vel) if not dir_vel is None else 'NULL' for dir_vel in [xvel, yvel, zvel])
                + "\n"
            )

        return self._move_iter

    def add_gravity(self, magnitude=9.8, xdir=0, ydir=0, zdir=-1):
        """
        Add gravity to the simulation
        Inputs:
        - magnitude: magnitude of the gravitational force
        - xdir, ydir, zdir: direction of the gravitational force
        """
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"\nfix grav all gravity {magnitude} vector {xdir} {ydir} {zdir}\n")
        pass

    def perturb(
        self,
        particles: Union[List[int], int] = None,
        type: Union[List[int],int] = None,
        magnitude=10 ** -5,
        xdir=0,
        ydir=0,
        zdir=0,
    ) -> int:
        """
        Add a force to EITHER a type or list of types of particles OR a particle or list of particles in the simulation
        Pass in EITHER:
        - particles: A particle or list of particles to perturb
        OR
        - type: A type or list of types of particle to perturb (output from methods like "add_grains")

        And:
        - magnitude: magnitude of the perturbative force
        - xdir, ydir, zdir: direction of the perturbative force
        """
        if not particles is None and not type is None:
            raise Exception("You can perturb EITHER a particle OR a type of particle")
        if particles is None and type is None:
            raise Exception("You must perturb either a particle or a type of particle")
        
        if not type is None:
            if isinstance(type,int):
                group = f"type {type}"
            else: 
                group = f"type " + " ".join([str(typ) for typ in type])
        else:
            if isinstance(particles,int):
                group = f"id {particles}"
            else: 
                group = f"id " + " ".join([str(part) for part in particles])

        self._perturb_iter += 1
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"\ngroup group_perturb_{self._perturb_iter} " + group + "\n")
            f.write(
                f"fix fix_perturb_{self._perturb_iter} group_perturb_{self._perturb_iter} gravity {magnitude} vector {xdir} {ydir} {zdir}\n"
            )

        return self._perturb_iter

    def add_viscosity(self, value, type: Union[List[int],int] = None, particles: Union[List[int],int] = None) -> int:
        """
        Add viscosity to all atoms or a set of types of atoms or a set of particles of atoms
        value - The strength of the viscosity
        type - The type or list of types of particles to which you want to add viscosity
        particles - The particle or list of particles to which you want to add viscosity

        If you pass in neither particles nor type, all particles in the simulation get viscous
        Returns: The id of the viscosity for if you want to remove this later
        """

        if not type is None and not particles is None:
            raise Exception("Can either add viscosity to types of particles or a list of particles, not both")
        
        if not type is None:
            if isinstance(type,int):
                group = f"type {type}"
            else: 
                group = f"type " + " ".join([str(typ) for typ in type])
        elif not particles is None:
            if isinstance(particles,int):
                group = f"id {particles}"
            else: 
                group = f"id " + " ".join([str(part) for part in particles])
        else: group = "all"

        self._visc_iter += 1
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write("\n")
            if group != "all":
                f.write(f"group group_viscosity_{self._visc_iter} " + group + "\n")
                f.write(
                    f"fix fix_viscosity_{self._visc_iter} group_viscosity_{self._visc_iter} viscous {value}\n"
                )
            else:
                f.write(
                    f"fix fix_viscosity_{self._visc_iter} all viscous {value}\n"
                )


        return self._visc_iter

    def remove_something(self, thing: str, id_of_that_thing: Union[List[int],int] = None):
        """
        This removes something from the simulation. Inputs:
        - thing: The kind of thing you want to remove, currently supported things are "viscosity", "particles",
            "type", "bonds", "angles" "gravity", "perturbation", "move", and "walls"
        - id_of_that_thing: The id of the thing you want to remove

        If thing is ___ then id_of_that_thing is ___ :
        - viscosity: the viscosity id
        - particles: a list of particle ids
        - type: a type of particles
        - gravity: don't pass in anything, just leave it set to None
        - perturbation: the id of the perturbation
        - move: the id of the move
        - walls: the id of the wall
        - bonds: if you pass in an int, id_of_that_thing is a type of bond. If you pass in a list, id_of_that_thing is a list of particles,
            the bonds between which will be deleted for example, if you pass in [1,2,3], all of the angles between particles 1,2, and 3 will be deleted.
            Be warned however, if particle 4 is bonded to particle 3, that bond will not be removed.
        - angles: if you pass in an int, id_of_that_thing is a type of angle. If you pass in a list, id_of_that_thing is a list of particles,
            the angles between which will be deleted, for example, if you pass in [1,2,3], all of the angles between particles 1,2, and 3 will be deleted.
            Be warned however, if particle 4 is angled to particle 3, that angle will not be removed.

        NOTE: If you
        """
        if thing not in [
            "viscosity",
            "particles",
            "type",
            "gravity",
            "perturbation",
            "move",
            "walls",
            "angles",
            "bonds"
        ]:
            raise Exception(
                "This is not something that I can remove, I can only remove either 'viscosity', 'particles','type', 'bonds', 'angles', 'perturbation, 'move', 'walls', or 'gravity' right now"
            )

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            if thing == "viscosity":
                f.write(f"unfix fix_viscosity_{id_of_that_thing}\n")
            if thing == "gravity":
                f.write(f"unfix grav\n")
            if thing == "perturbation":
                f.write(f"unfix fix_perturb_{id_of_that_thing}\n")
            if thing == "move":
                f.write(f"unfix fix_move_{id_of_that_thing}\n")
            if thing == "type":
                index_drop = self._particles[ self._particles['type'] == id_of_that_thing ].index
                self._particles.drop(index_drop , inplace=True)
                f.write(f"group group_delete type {id_of_that_thing}\n")
                f.write(f"delete_atoms group group_delete\n")
                f.write(f"group group_delete delete\n")
            if thing == "particles":
                self._particles.drop(index=id_of_that_thing, inplace=True)
                if isinstance(id_of_that_thing,int):
                    f.write(f"group group_delete id {id_of_that_thing}\n")
                else:
                    f.write(f"group group_delete id " + " ".join(str(part) for part in id_of_that_thing) + "\n")
                f.write(f"delete_bonds group_delete multi any\n")
                f.write(f"delete_atoms group group_delete\n")
                f.write(f"group group_delete delete\n")
            if thing == "bonds":
                if isinstance(id_of_that_thing,int):
                    f.write(f"delete_bonds all bond {id_of_that_thing}\n")
                else:
                    f.write(f"group group_delete id " + " ".join(str(part) for part in id_of_that_thing) + "\n")
                    f.write(f"delete_bonds group_delete bond 1*\n")
                    f.write(f"group group_delete delete\n")
            if thing == "angles":
                if isinstance(id_of_that_thing,int):
                    f.write(f"delete_bonds all angle {id_of_that_thing}\n")
                else:
                    f.write(f"group group_delete id " + " ".join(str(part) for part in id_of_that_thing) + "\n")
                    f.write(f"delete_bonds group_delete angle *\n")
                    f.write(f"group group_delete delete\n")
            if thing == "walls":
                if self._walls[id_of_that_thing - 1][1]:
                    f.write(f"unfix fix_walls_{id_of_that_thing}\n")
                f.write(f"region region_{id_of_that_thing} delete\n")
                if self._walls[id_of_that_thing - 1][0]:
                    f.write(f"group group_wall_{id_of_that_thing} delete\n")
        pass

    def custom(self, statement: str):
        """
        Add a custom statement to the simulation
        - statement: The line that you want to add
        """
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(statement + "\n")

        pass

    def design_thermo(
        self,
        thermo_every_this_many_seconds: float = None,
        thermo_list: List[str] = None,
    ):
        """
        Design the mid-simulation output to the log.lammps file. Inputs:
        - thermo_every_this_many_seconds: Output the thermo every this many seconds
        - thermo_list: A list of things you want in the thermo output
        """
        self._thermo_step = round(thermo_every_this_many_seconds / self._timestep)
        if not thermo_list is None:
            self._thermo_list = thermo_list
        else:
            self._thermo_list = ["step", "atoms", "time"]
        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write("\n")
            f.write("thermo_style custom " + " ".join(self._thermo_list) + "\n")
            f.write("thermo_modify lost warn\n")
            f.write(f"thermo {self._thermo_step}\n")

        pass

    def design_dump_files(
        self, dump_file_every_this_many_seconds: float, dump_list: List[str] = None
    ):
        """
        This command makes it such that we output dump files, and it also decides what will be in those dump files. 
        If we have already called this function once in the script, then this does not re-design the dump files,
        rather, it simply allows you to change how often those dump files will be output.
        Inputs:
        - dump_file_every_this_many_seconds: This sets how many seconds the simulation waits between printing
            dump files.
        - dump_list: This determines what will be in the dump file.
        
        The way that I have this set up, we always include the id, type, and radius of the particles, as well
        as their x y z positions. If you pass in an empty list here then that is all you'll get in your dump files.
        If you leave this set to None, we will also include the x, y, and z force components, as well as the bending
        and stretching energy. You can also additionally pass in 'contacts', which will give you the number of contacts
        per particle, 'pressure', which will output the pressure on each particle, or anything else that LAMMPS will
        accept in a custom dump file (https://docs.lammps.org/dump.html, the section on 'custom' attributes)
        """

        if self._dump_file_every_this_many_seconds:
            self._dump_file_every_this_many_seconds = dump_file_every_this_many_seconds
        
        else:
            self._dump_file_every_this_many_seconds = dump_file_every_this_many_seconds
            if not dump_list is None:
                self._dump_list = dump_list
            else:
                self._dump_list = ["fx", "fy", "fz", "bending_energy", "stretching_energy"]

            with open(os.path.join(self._path, "in.main_file"), "a") as f:
                f.write("\n")
                if "contacts" in self._dump_list:
                    f.write("compute contacts all contact/atom\n")
                    self._dump_list.remove("contacts")
                    self._dump_list.append("v_contacts")
                if "pressure" in self._dump_list:
                    f.write("compute stress all stress/atom NULL\n")
                    f.write("compute contacts all contact/atom\n")
                    f.write(
                        "variable pressure atom 2*(c_stress[1]+c_stress[2]+c_stress[3])/(c_contacts+.001)\n"
                    )
                    self._dump_list.remove("pressure")
                    self._dump_list.append("v_pressure")
                if "bending_energy" in self._dump_list:
                    f.write("compute bendingE all pe/atom angle\n")
                    self._dump_list.remove("bending_energy")
                    self._dump_list.append("c_bendingE")
                if "stretching_energy" in self._dump_list:
                    f.write("compute stretchingE all pe/atom bond\n")
                    self._dump_list.remove("stretching_energy")
                    self._dump_list.append("c_stretchingE")

                f.write(
                    f"dump pump all custom 1 out*.dump id type radius x y z "
                    + " ".join(str(item) for item in self._dump_list)
                    + "\n"
                )
                f.write("dump_modify pump pad 11\n")
        pass

    def run_simulation(self, time: float, timestep: float = None):
        """
        Run the simulation for a given amount of time. Inputs:
        - time: Amount of time to run the simulation for
        - timestep: The intended timestep of the simulation.

        The auto-generated timestep is currently in-production (TODO), when it is done, read the following:
        A note on this, a timestep is already estimated automatically via this script, and if 
        you don't pass anything into the timestep value, the simulation will use this automatically 
        calculated timestep. However, there will probably be plenty of situations where you will 
        need to change this timestep manually. If the timestep that the script automatically calculates
        is too small, the simulations will take a long time. If instead the auto-generated timestep is
        too large, the simulation will be unstable, and you might have your particles flying all over the
        place (resulting in errors where lammps tells you that it can no longer find the atoms in a bond or angle).
        If either of these things are happening to you, you might want to manually change the timestep. In that
        case, the auto-generated timestep can be a good jumping-off point!

        Another note, LAMMPS will not let you re-set the timestep if you have already run some of the simulation, 
        and then have applied a fix move. That is, if you have already simulated something -- called the 
        run_simulation() method -- and then you call the move() method, LAMMPS freaks out, because it bases its
        movement calculation on the timestep. If you change the timestep, that calculation is now out of whack.
        This can also happen if you have not run anything, but instead have called a function that runs something.
        For example, if you call `connect_particles_to_elastic` before you set the timestep, then you move something,
        and THEN you try to set the timestep, LAMMPS also freaks out, because inside of `connect_particles_to_elastic`
        I have some `run 0`s. That is, I have to execute the run command within that function to get it to work
        properly. SO, if you haven't run any of the simulation, AND you want to `connect_particles_to_elastic`,
        AND you want to fix move, you will have to `manually_edit_timestep` before you call the `move` command

        Based on this, if you have both:
        - Already run some of the simulation, AND
        - called the move() command

        OR
        - connected particles to an elastic material
        - called the move() command
        Then this method will not allow you to reset the timestep
        """
        if not self._timestep:
            raise Exception("It seems like we don't have any particles to simulate!")

        if timestep is None:
            timestep = self._timestep

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            if self._have_run and self._move_iter>0:
                print("\n######")
                print("Cant reset timestep after sim has run and `fix move` has been performed (move() has been called)")
                print("Ignoring the timestep and running simulation with the last timestep that was set")
                print("######\n")
                number_of_timesteps_run = round(time / self._manual_timestep)
                number_of_timesteps_dump = round(self._dump_file_every_this_many_seconds / self._manual_timestep)
            else:
                f.write(f"\ntimestep {timestep}\n")
                number_of_timesteps_run = round(time / timestep)
                number_of_timesteps_dump = round(self._dump_file_every_this_many_seconds / timestep)
                self._manual_timestep = timestep
            if self._dump_file_every_this_many_seconds:
                f.write(
                    f"dump_modify pump every {number_of_timesteps_dump}\n"
                )
            f.write(f"run {number_of_timesteps_run}\n")
            self._have_run = True

        pass

    def manually_edit_timestep(self,timestep: float):
        """
        First thing you need to know about this method, you usually set the timestep when you run run_simulation(),
        so more often than not, this method is not needed. However, there are some rare cases where you might
        want to set this manually. For example, lets say I'm about to apply a fix move (call the move() method),
        but I also want to change the timestep before this happens. I won't be able to change the timestep after
        the fix move because of what I mention in the desription of run_simulation(). Therefore, you can use
        this function to slip in a timestep change right before the fix move.
        """

        if self._have_run and self._move_iter>0:
            raise Exception("Cant change the timestep if you have started the simulation and applied a fix move")

        with open(os.path.join(self._path, "in.main_file"), "a") as f:
            f.write(f"\ntimestep {timestep}\n")
            self._manual_timestep = timestep
    
        pass


def _dist(coords, i, j):
    return ((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2)**0.5
