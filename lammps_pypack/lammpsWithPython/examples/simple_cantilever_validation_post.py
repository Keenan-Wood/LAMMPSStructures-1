import os
import pathlib

import numpy as np

import matplotlib.pyplot as plt

simname = "simple_cantilever_validation"
sim_path = pathlib.Path(__file__).parent.resolve()

timestep = 10 ** -7

# Build list of calculated z values of end node
t_sim = []
z_sim = []
F_sim = []
directory_str = str(sim_path) + f'/{simname}/raw/'
directory = os.fsencode(directory_str)
for file in os.listdir(directory):
    atoms_def = False
    filename = os.fsdecode(file)
    if filename.endswith(".dump"):
        with open(directory_str + filename) as f:
            for i_line, line in enumerate(f):
                data = line.rstrip('\n').split(' ')
                if i_line == 1:
                    t_sim.append(float(data[0]) * timestep)
                if data[0] == 'ITEM:' and data[1] == 'ATOMS':
                    atoms_def = True
                    continue
                if atoms_def == True and data[0] == '2':
                    z_sim.append(float(data[5]))
                    F_sim.append(float(data[8]))
                    break
        continue
    else:
        continue

t_sim, z_sim, F_sim = (list(t) for t in zip(*sorted(zip(t_sim, z_sim, F_sim))))

# Analytic solution
beam_length = 0.1
beam_thickness = 0.002
E_beam = 0.96 * 10 ** 6
I = np.pi/4 * (beam_thickness/2)**4

F_sim = np.array(F_sim)
z_sim = np.array(z_sim)

F_calc = - z_sim / beam_length**3 * 3*E_beam*I
F_error = (F_calc - F_sim) / F_calc
#z_calc = - F_sim * beam_length**3 / (3*E_beam*I)

plt.figure(figsize=(8, 6))
plt.plot(z_sim, F_sim, 'ro-', label='simulated')
plt.plot(z_sim, F_calc, 'b--', label='analytic')
plt.xlabel("Displacement Applied at End")
plt.ylabel("Resulting Force")
plt.title("Simulated vs Analytic Results")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot image\n",
img_fname = directory_str + "analytic-comparison.png"
plt.savefig(str(img_fname))

# Plot Percent Error
plt.figure(figsize=(8, 6))
plt.plot(z_sim, F_error, 'ro-', label='percent error')
plt.xlabel("Displacement Applied at End")
plt.ylabel("Resulting Force Percent Error")
plt.title("Simulated vs Analytic Percent Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot image\n",
img_fname = directory_str + "analytic-comparison-error.png"
plt.savefig(str(img_fname))