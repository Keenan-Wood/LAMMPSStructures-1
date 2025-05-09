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

simname = "simple_cantilever_validation"

beam_length = 0.1
beam_thickness = 0.002
Np_beam = 150
Np_side = 20
squish_factor = 0.1
E_beams = 0.96 * 10 ** 6
E_walls = 10 ** 4

density = 0.5
viscosity = 2 * 10 ** -7 #* 10**3
timestep = 1 * 10 ** -7
dump_timestep = 10 ** -2
simtime = 1
#simtime = 0.001

# Start a simulation with the name simname
sim_path = pathlib.Path(__file__).parent.resolve()
sim = lsim.Simulation(simname, 3, beam_length + 0.01, beam_thickness + 0.01, 0.05, sim_dir = sim_path)
# Make the simulationStation hard. We can also do this sim periodically, so this is not required
sim.add_walls(youngs_modulus = E_walls)
sim.turn_on_granular_potential(youngs_modulus = E_walls)

## Define the nodes with coordinates and diameters
# Here the coordinates of node 1 and 3 will be calculated from the parametric equations we provide the elements
node_diameter = beam_thickness
nodes = [
    ([-beam_length/2, 0, 0], node_diameter),
    ([beam_length/2, 0, 0], node_diameter)
    ]
constraints = [[0, 1,1,1]]

(E, rho) = (E_beams, density)
materials = [['test_material_0', E, rho]]
xsecs = [[0, beam_thickness]]

# Define the elements with node id pairs
elements = [(0, 1)]

# Create the structure object
new_structure = lstruct.Structure(
    node_list = nodes,
    material_list = materials,
    xsection_list = xsecs,
    element_list = elements,
    constraint_list = constraints)
new_structure.plot(str(sim_path) + f'/{simname}/', 'structure_1.png')

# Discretize the structure to generate list of atoms and bonds (and their types)
(atom_type_list, bond_type_list, atoms, bonds) = new_structure.discretize(beam_length/Np_beam)

# Add structure atoms to simulation, apply constraints, and add bond types, bonds
sim.add_atoms(structure=new_structure)
sim.apply_node_constraints(new_structure.nodes)
# Clamp particle next to fixed node
sim.move(particles = new_structure.elements[0].atoms[0].id, xvel = 0, yvel = 0, zvel = 0)
sim.add_bond_types(structure=new_structure)
sim.add_bonds(structure=new_structure)

# Actuate downward
atoms_to_move = [new_structure.nodes[1].atom.id]
sim.move(particles = atoms_to_move, xvel = 0, yvel = 0, zvel = -squish_factor * beam_length / simtime)

# Add the viscosity for energy dissipation
sim.add_viscosity(viscosity)

# Make the dump files and run the simulation
sim.design_dump_files(dump_timestep)
sim.run_simulation(simtime, timestep)

# Call lammps to run simulation
lsim.run_lammps(sim)

# Render dump files with Ovito
img_size = (640, 480)
lrend.render_dumps(img_size, str(sim_path) + f'/{simname}', orient_name=['front'])