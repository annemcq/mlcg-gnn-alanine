"""
Generate a reference all-atom trajectory of alanine dipeptide (ACE-ALA-NME)
in implicit solvent using OpenMM, saving positions and forces at regular
intervals for later use in coarse-graining (force matching).
"""
import numpy as np
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff, HBonds
from openmm import LangevinMiddleIntegrator, unit
import time

pdb = PDBFile('data/alanine-dipeptide.pdb')
forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                  constraints=HBonds)

temperature = 300 * unit.kelvin
friction = 1.0 / unit.picosecond
timestep = 2.0 * unit.femtosecond

integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

# equilibrate
simulation.step(5000)

n_frames = 20000
report_interval = 50  # save every 50 steps (100 fs) -> ~2 ns trajectory total
positions_nm = np.zeros((n_frames, system.getNumParticles(), 3))
forces_kjmolnm = np.zeros((n_frames, system.getNumParticles(), 3))

t0 = time.time()
for i in range(n_frames):
    simulation.step(report_interval)
    state = simulation.context.getState(getPositions=True, getForces=True)
    positions_nm[i] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    forces_kjmolnm[i] = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    if (i+1) % 2000 == 0:
        print(f"{i+1}/{n_frames} frames, {time.time()-t0:.1f}s elapsed")

np.save('data/aa_positions_nm.npy', positions_nm)
np.save('data/aa_forces_kjmolnm.npy', forces_kjmolnm)

atom_names = [a.name for a in pdb.topology.atoms()]
residue_names = [a.residue.name for a in pdb.topology.atoms()]
np.save('data/atom_names.npy', np.array(atom_names))
np.save('data/residue_names.npy', np.array(residue_names))

print("done. total time:", time.time()-t0, "s")
print("positions shape:", positions_nm.shape)
