import os
import pathlib
import sys
import inspect
import subprocess
import numpy as np
import pandas as pd

from mpi4py import MPI
from lammps import lammps

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import lammpsWithPython.lammps_structure as lstruct
import lammpsWithPython.lammps_simulation as lsim
import lammpsWithPython.lammps_render as lrend

simname = "truss"

n_beams = 10
d_between_beams = 0.02
beam_length = 0.02
beam_thickness = 0.002
Np_beam = 20
Np_side = 20
squish_factor = 0.01
E_beams = 0.96 * 10 ** 6
E_walls = 10 ** 4

density = 0.5
viscosity = 2 * 10 ** -7 * 10**4
timestep = 1 * 10 ** -7
dump_timestep = 10 ** -2
simtime = 1

Np_between = int(Np_beam / beam_length * d_between_beams) - 1
x_rng = 0.5*d_between_beams*(n_beams - 1) * np.array([-1, 1])
dy = beam_length / (Np_beam - 1)

# Start a simulation with the name simname
sim_path = pathlib.Path(__file__).parent.resolve()
sim = lsim.Simulation(simname, 3, d_between_beams*n_beams + 0.02, beam_thickness + 0.01, beam_length + 0.01, sim_dir = sim_path)
# Make the simulationStation hard. We can also do this sim periodically, so this is not required
sim.add_walls(youngs_modulus = E_walls)
sim.turn_on_granular_potential(youngs_modulus = E_walls)

## Define the nodes with coordinates and diameters
# Here the coordinates of node 1 and 3 will be calculated from the parametric equations we provide the elements
node_diameter = beam_thickness
nodes = [
    ([0, 0, 0], node_diameter),
    ([0, 0, beam_length], node_diameter),
    ([d_between_beams, 0, 0], node_diameter),
    ([d_between_beams, 0, beam_length], node_diameter),
    ]
constraints = None

(E, rho) = (E_beams, density)
materials = [['test_material_0', E, rho]]
xsecs = [[0, beam_thickness]]

# Define the elements with rnode id pairs
elements = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3)]

# Create the structure object
new_structure = lstruct.Structure(
    node_list = nodes,
    material_list = materials,
    xsection_list = xsecs,
    element_list = elements,
    constraint_list = constraints)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1.png')

# Translate structure to place the origin in the center (after patterning)
new_structure.translate([-d_between_beams*n_beams/2, 0, -beam_length/2], copy=False)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1_shifted.png')

# Join an offset copy of the structure
new_structure = new_structure.pattern_linear(np.array([1,0,0]), 1, offset = d_between_beams)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1_dual-cell.png')

# Pattern the structure (repeat n_beams - 1 times in x-direction, and join together)
new_structure = new_structure.pattern_linear(np.array([1,0,0]), n_beams - 3, offset = d_between_beams)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1_patterned.png')

# Discretize the structure to generate list of atoms and bonds (and their types)
new_structure.discretize(beam_length/Np_beam)

# Add end bonds
#end_node_id = len(new_structure.nodes) - 1
#new_structure.add_node_bonds([end_node_id - 1, end_node_id, end_node_id - 2], ('angle', 'cosine/delta'), ['angular_stiffness', 'rest_plane_angle'])
#new_structure.add_node_bonds([end_node_id, end_node_id - 1, end_node_id - 3], ('angle', 'cosine/delta'), ['angular_stiffness', 'rest_plane_angle'])

# Add structure atoms to simulation, apply constraints, and add bond types, bonds
sim.add_atoms(structure=new_structure)
constraint_list = [[0, 1,1,1], [1, 1,1,1]]
new_structure.add_constraints(constraint_list)
sim.apply_node_constraints(new_structure.nodes)
sim.add_bond_types(structure=new_structure)
sim.add_bonds(structure=new_structure)

# Move end nodes down
moving_nodes = [2*n_beams - 2]
atoms_to_move = [new_structure.nodes[i_nd].atom.id for i_nd in moving_nodes]
#atoms_to_move += [atm.id for el in new_structure.elements for atm in el.atoms if el.node_a.id in moving_nodes and el.node_b.id in moving_nodes]
sim.move(particles = atoms_to_move, xvel = 0, yvel = 0, zvel = -squish_factor * beam_length)

# Clamp all atoms of fixed top
#fixed_nodes_list = [2*i_beam for i_beam in range(n_beams)]
#fixed_element_list = [el for el in new_structure.elements if el.node_a.id in fixed_nodes_list and el.node_b.id in fixed_nodes_list]
#for el in fixed_element_list:
#    atom_ids = [atm.id for atm in el.atoms]
#    sim.move(particles = atom_ids, xvel = 0, yvel = 0, zvel = 0)

# Add viscosity for energy dissipation
sim.add_viscosity(viscosity)

# Add gravity
# sim.add_gravity()

# Make the dump files and run the simulation
sim.design_dump_files(dump_timestep)
sim.run_simulation(simtime, timestep)

# Call lammps to run simulation
lsim.run_lammps(sim)

# Render dump files with Ovito
img_size = (640, 480)
lrend.render_dumps(img_size, str(sim_path) + f'/{simname}', orient_name=['front'])

# Stitch gifs into one composite gif
#gif_fnames = ['ovito_anim_front.gif', 'ovito_anim_perspective.gif']
#lrend.stitch_gifs(str(sim_path) + f'/{simname}', gif_fnames, (1,2), [0,1])