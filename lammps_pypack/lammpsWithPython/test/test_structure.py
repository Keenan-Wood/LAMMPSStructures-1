"""\
This script provides an example of using lammpsWithPython and Ovito (with python)
 - Based on mushroom example found here: https://github.com/adguerra/LAMMPSStructures/blob/main/examples/mushroom/mushroom.py
"""

import os
import subprocess
import numpy as np
import pandas as pd

from mpi4py import MPI
from lammps import lammps
from lammpsWithPython.lammps_object import *

from lammpsWithPython.lammps_structure import *
import runLammps as rL
import ovitoView as ov

simname = "simExample_4"

n_beams = 2
d_between_beams = 0.005
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
#simtime = 1
simtime = 0.1

Np_between = int(Np_beam / beam_length * d_between_beams) - 1
Np_hori = 2*Np_side + Np_between + 2
x_rng = 0.5*d_between_beams*(n_beams - 1) * np.array([-1, 1])
dy = beam_length / (Np_beam - 1)

# Start a simulation with the name simname
#sim = Simulation(simname, 3, Np_hori*dy + 0.02, 0.01, 0.11)
sim = Simulation(simname, 3, 10, 0.5, 1)
# Make the simulationStation hard. We can also do this sim periodically, so this is not required
sim.add_walls(youngs_modulus = E_walls)

# We're gonna do something tricky, which is we will turn on all granular interactions between all beams, and
# Then in the loop where we add the beams we will turn off the granular interactions between the beams and themselves
# If the particles in the beams are granular with themselves, it will mess up the beam mechanics
sim.turn_on_granular_potential(youngs_modulus = E_walls)


nodes = [[0.1*x_i, 0, 0] for x_i in range(5)] + [[] for _ in range(5)]
(E, rho) = (0.96 * 10**6, 0.5)
materials = [['test_material_0', E, rho]]
beam_thickness = 0.002
xsecs = [[0, beam_thickness]]
r_helix = 0.01
param_eqs = [lambda t: r_helix*(np.cos(5*2*np.pi*t) - 1),
             lambda t: r_helix*np.sin(5*2*np.pi*t),
             lambda t: 4*r_helix*t]
elements = [[x_i, x_i + 1] for x_i in range(4)]
elements += [[x_i, x_i + 5, 'test_material_0', 0, param_eqs] for x_i in range(5)]
elements += [[x_i, x_i + 1] for x_i in range(5, 9)]
connections = [[[0, 1, 2], 'dihedral', None], [[5, 6, 7], 'dihedral', None]]
constraints = [[0, 1,1,1,1,1,1], [4, 1,1,1,1,1,1]]
new_frame = frame(
    nodes, material_list = materials,
    xsection_list = xsecs,
    element_list = elements,
    element_connections = connections,
    constraint_list = constraints)
(atom_type_list, bond_type_list, atoms, bonds) = new_frame.discretize(0.02)

sim.add_atoms(atom_type_list, atoms, "structure_1.txt")
sim.apply_node_constraints(new_frame.nodes)
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
sim.move(particles = new_frame.nodes[6].atom.id, xvel = 0, yvel = 0, zvel = - 2 * r_helix / simtime)

# Actuate rightward
#mid_point = [int(Np_beam/2)]
#sim.move(particles = mid_point, xvel = 0.1 * squish_factor * beam_length / simtime, yvel = 0, zvel = 0)

# Perturb the beams to buckle to the left or right randomly
#dirs = np.random.rand(len(beam_positions),1)
#p1 = sim.perturb(type = [i+1 for i in np.where(dirs>0.5)[0].tolist()],xdir = 1)
#p2 = sim.perturb(type = [i+1 for i in np.where(dirs<=0.5)[0].tolist()],xdir = -1)

# Add the viscosity, which just helps the simulation from exploding, mimics the normal 
# damping of air and slight viscoelasticity which we live in but don't appreciate
sim.add_viscosity(viscosity)

# Make the dump files and run the simulation
sim.design_dump_files(0.01)
sim.run_simulation(simtime, timestep)

# Call lammps to run simulation
sim_path = os.getcwd() + "/" + simname + "/"
sim_type = "serial"
rL.run_lammps(sim_type, sim_path)
#try:
#    subprocess.check_call(["mpirun", "-np", str(os.environ['NSLOTS']), "python3", "src/runLammps.py", "\""+sim_type+"\"", "\""+sim_path+"\""])
#except subprocess.CalledProcessError as e:
#    print(e)
#subprocess.call(cmd_1 + cmd_2, shell=True)

# Render dump files with Ovito
img_size = (640, 480)
ov.render_dumps(img_size, sim_path)

# Stitch gifs into one composite gif
gif_fnames = ['ovito_anim_front.gif', 'ovito_anim_perspective.gif']
ov.stitch_gifs(sim_path, gif_fnames, (1,2), [0,1])