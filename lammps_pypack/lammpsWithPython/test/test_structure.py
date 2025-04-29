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

import lammps_structure as lstruct
import lammps_simulation as lsim
import lammps_render as lrend

simname = "test_structure"

n_beams = 5
d_between_beams = 0.02
beam_length = 0.100
beam_thickness = 0.002
Np_beam = 100
Np_side = 20
squish_factor = 0.2
E_beams = 0.96 * 10 ** 6
E_walls = 10 ** 4

density = 0.5
viscosity = 2 * 10 ** -7
timestep = 1 * 10 ** -7
dump_timestep = 10 ** -7
simtime = 1
#simtime = 0.001

Np_between = int(Np_beam / beam_length * d_between_beams) - 1
Np_hori = 2*Np_side + Np_between + 2
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
node_diameter = 0.002
nodes = [
    ([0, 0, 0], node_diameter),
    (None, node_diameter),
    ([d_between_beams, 0, 0], node_diameter),
    (None, node_diameter)
    ]
constraints = [[1, 1,1,1,1,1,1], [3, 1,1,1,1,1,1]]

(E, rho) = (E_beams, density)
materials = [['test_material_0', E, rho]]
xsecs = [[0, beam_thickness]]

# Define the elements with node id pairs
r_helix = 0.003
N_turns = 10
param_eqs = [lambda t: r_helix*(np.cos(N_turns*2*np.pi*t) - 1),
             lambda t: r_helix*np.sin(N_turns*2*np.pi*t),
             lambda t: beam_length*t]
elements = [(0, 1, param_eqs), (2, 3, param_eqs), (0, 2), (1, 3)]
connections = [
    ([0, 2, 4], ('dihedral','spherical'), None),
    ([1, 3, 5], ('dihedral','spherical'), None),
    ]

# Create the structure object
new_structure = lstruct.Structure(
    node_list = nodes,
    material_list = materials,
    xsection_list = xsecs,
    element_list = elements,
    connection_list = connections,
    constraint_list = constraints)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1.png')

# Translate structure to place the origin in the center (after patterning)
new_structure.translate([-d_between_beams*n_beams/2, 0, -beam_length/2], copy=False)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1_shifted.png')

# Pattern the structure (repeat n_beams - 1 times in x-direction, and join together)
new_structure = new_structure.pattern_linear(np.array([1,0,0]), n_beams - 1, offset = d_between_beams)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1_patterned.png')

# Discretize the structure to generate list of atoms and bonds (and their types)
(atom_type_list, bond_type_list, atoms, bonds) = new_structure.discretize(beam_length/Np_beam)

# Add node atoms to simulation
node_atom_types = atom_type_list[0:len(new_structure.nodes)]
sim.add_atoms(node_atom_types, atoms, "node")

# Add element atoms to simulation
element_atom_types = atom_type_list[len(new_structure.nodes):]
sim.add_atoms(element_atom_types, atoms, "element")
# Turn of granular interaction between atoms belonging to the same element
for el_atm_type in element_atom_types:
    sim.turn_on_granular_potential(type1 = el_atm_type[0], type2 = el_atm_type[0], youngs_modulus = 0)

sim.apply_node_constraints(new_structure.nodes)
sim.add_bond_types(bond_type_list)
sim.add_bonds(bond_type_list, bonds)
sim.turn_on_granular_potential(youngs_modulus = 0)

#for beam_dat in beam_data:
#    beam, _, _ = sim.add_beam(beam_dat[0], beam_dat[1], beam_dat[2], beam_thickness, E_beams, density)
#    sim.turn_on_granular_potential(type1 = beam, type2 = beam, youngs_modulus = 0)

#bond_stiffness = E_beams * beam_thickness**2 / (2*dy)
#for bond_dat in bond_data:
#    _ = sim.construct_many_bonds(bond_dat, bond_stiffness, dy)

#angle_stiffness = E_beams * (beam_thickness ** 4) / (12*dy)
#for angle_dat in angle_data:
#    _ = sim.construct_many_angles(angle_dat, angle_stiffness)

#x_end_list = np.linspace(x_rng[0], x_rng[1], n_beams).tolist()
#for x_end in x_end_list:
#    z_end = np.array([-beam_length/2, beam_length/2])
#    beam, _, _ = sim.add_beam(Np_beam, np.array([x_end,0,z_end[0]]), np.array([x_end,0,z_end[1]]), beam_thickness, E_beams, density)
#    sim.turn_on_granular_potential(type1 = beam, type2 = beam, youngs_modulus = 0)

#z_end_list_hori = [-beam_length/2 - dy, beam_length/2 + dy]
#x_end = np.array([x_rng[0] - dy*(Np_side - 1), x_rng[1] + dy*(Np_side + 1)])
#for z_end in z_end_list_hori:
#    hori_beam, _, _ = sim.add_beam(Np_hori, np.array([x_end[0],0,z_end]), np.array([x_end[1],0,z_end]), beam_thickness, E_beams, density)
#    sim.turn_on_granular_potential(type1 = hori_beam, type2 = hori_beam, youngs_modulus = 0)

#hori_joint_ids = np.array([n_beams*Np_beam + Np_side + 1,
#                           n_beams*Np_beam + Np_side + Np_between + 2,
#                           n_beams*Np_beam + Np_hori + Np_side + 1,
#                           n_beams*Np_beam + Np_hori + Np_side + Np_between + 2])
#beam_end_pairs = np.array([[1, hori_joint_ids[0]], [Np_beam, hori_joint_ids[2]],
#                           [Np_beam + 1, hori_joint_ids[1]], [2*Np_beam, hori_joint_ids[3]]])
#beam_end_triplets = np.array([[hori_joint_ids[0] + 1, hori_joint_ids[0], 1],
#                              [hori_joint_ids[2] + 1, hori_joint_ids[2], Np_beam],
#                              [hori_joint_ids[1] - 1, hori_joint_ids[1], Np_beam + 1],
#                              [hori_joint_ids[2] - 1, hori_joint_ids[2], 2*Np_beam]])
#bond_stiffness = E_beams * beam_thickness**2 / (2*dy)
#angle_stiffness = E_beams * (beam_thickness ** 4) / (12*dy)
#_ = sim.construct_many_bonds(beam_end_pairs, bond_stiffness, dy)
#_ = sim.construct_many_angles(beam_end_triplets, angle_stiffness)

# Clamp the clamp particles. Here we also clamp them in x and y, but this can be changed.
#bottom_clamp = [n_beams*Np_beam + 1, n_beams*Np_beam + 2, n_beams*Np_beam + Np_hori - 1, n_beams*Np_beam + Np_hori]
#top_clamp = [n_beams*Np_beam + Np_hori + 1, n_beams*Np_beam + Np_hori + 2, n_beams*Np_beam + 2*Np_hori - 1, n_beams*Np_beam + 2*Np_hori]
#sim.move(particles = bottom_clamp, xvel = 0, yvel = 0, zvel = 0) #zvel = squish_factor * beam_length / simtime)
#sim.move(particles = top_clamp, xvel = 0, yvel = 0, zvel = 0) #zvel = - squish_factor * beam_length / simtime)

# Actuate downward
#center_point = [int(n_beams*Np_beam + Np_hori + Np_hori/2)]
#sim.move(particles = center_point, xvel = 0, yvel = 0, zvel = -squish_factor * beam_length / simtime)
atoms_to_move = [new_structure.nodes[i_nd].atom.id for i_nd in range(n_beams, 2*n_beams)]
sim.move(particles = atoms_to_move, xvel = 0, yvel = 0, zvel = 0.2 * beam_length / simtime)

# Actuate rightward
#mid_point = [int(Np_beam/2)]
#sim.move(particles = mid_point, xvel = 0.1 * squish_factor * beam_length / simtime, yvel = 0, zvel = 0)

# Perturb the beams to buckle to the left or right randomly
#dirs = np.random.rand(len(beam_positions),1)
#p1 = sim.perturb(type = [i+1 for i in np.where(dirs>0.5)[0].tolist()],xdir = 1)
#p2 = sim.perturb(type = [i+1 for i in np.where(dirs<=0.5)[0].tolist()],xdir = -1)
mid_atom_ids = [new_structure.get_atom_id([2*i_nd, 2*i_nd + 1], 0.5) for i_nd in range(n_beams)]
sim.perturb(particles = mid_atom_ids, xdir = 1)

# Add the viscosity, which just helps the simulation from exploding, mimics the normal 
# damping of air and slight viscoelasticity which we live in but don't appreciate
sim.add_viscosity(viscosity)

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