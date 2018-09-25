import os
import posixpath
import neuron
import LFPy
from glob import glob
import yaml
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sys
import random
import MEAutility as mea

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")


def posixpth(pth):
    """
    Replace Windows path separators with posix style separators
    """
    return pth.replace(os.sep, posixpath.sep)


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within
    Arguments
    ---------
    f : file, mode 'r'
    Returns
    -------
    templatename : str
    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print('template {} found!'.format(templatename))
            continue
    return templatename


def get_sections_number(cell):
    '''
    Returns the number of different cell sections and their names.
    '''
    nlist = []
    for i, name in enumerate(cell.allsecnames):
        if name != 'soma':
            name = name[:name.rfind('[')]

        if name not in nlist:
            nlist.append(name)
    n = len(nlist)
    return n, nlist


CWD = os.getcwd()
compilation_folder = 'morphologies/hoc_combos_syn.1_0_10.allmods'  # change to the directory containing compiled .mod files

NRN = "morphologies/hoc_combos_syn.1_0_10.allzips/L23_PC_cADpyr229_5"  # change to whatever Blue Brain project neuron models you want

neuron.load_mechanisms(compilation_folder)
os.chdir(NRN)

# PARAMETERS
tstop = 400  # [ms] sim duration
dt = 2**-6  # time step
n_tsteps = int(tstop / dt + 1)
t = np.arange(n_tsteps) * dt

# Current Pulse configuration, monophasic
pulse_start = int(200. / dt + 1)
pulse_duration = int(.5 / dt + 1)
amp = 50 * 10**3  # [uA]
pulse = np.zeros(n_tsteps)
pulse[pulse_start:(pulse_start + pulse_duration)] = 1.
sigma = .3

# MEA creation
name_shape_ecog = 'bisc_12'  # NEED TO ADD bisc_12 by default
bisc = mea.return_mea(name_shape_ecog)
bisc.points_per_electrode = 100
pia_height = 50  # [um]

bisc.set_currents(amp * np.ones(bisc.get_electrodes_number()))
bisc.move([0, 0, pia_height])

# output folder
output_f = CWD  # Savefig disabled by default, uncomment the call at end of file

# get the template name
f = open("template.hoc", 'r')
templatename = get_templatename(f)
f.close()

# get biophys template name
f = open("biophysics.hoc", 'r')
biophysics = get_templatename(f)
f.close()

# get morphology template name
f = open("morphology.hoc", 'r')
morphology = get_templatename(f)
f.close()

# get synapses template name
f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
synapses = get_templatename(f)
f.close()

print('Loading constants')
neuron.h.load_file('constants.hoc')

if not hasattr(neuron.h, morphology):
    """Create the cell model"""
    # Load morphology
    neuron.h.load_file(1, "morphology.hoc")
if not hasattr(neuron.h, biophysics):
    # Load biophysics
    neuron.h.load_file(1, "biophysics.hoc")
    print("biophysics loaded")
if not hasattr(neuron.h, templatename):
    # Load main cell template
    neuron.h.load_file(1, "template.hoc")

add_synapses = False

morphologyfile = glob(os.path.join('morphology', '*'))

LFPy.cell.neuron.init()


cell = LFPy.TemplateCell(morphology=morphologyfile[0],
                         templatefile=posixpth(os.path.join(NRN, 'template.hoc')),
                         templatename=templatename,
                         templateargs=1 if add_synapses else 0,
                         tstop=tstop,
                         dt=dt,
                         extracellular=True,
                         nsegs_method=None,
                         passive=False)

# Set Position and Rotation
cell.set_rotation(x=np.pi / 2, z=np.pi / 2)

if np.max(cell.zend) > 0:
    cell.set_pos(z=-np.max(cell.zend) - 50)

source_amps = bisc.currents

positions_x = np.zeros(bisc.get_electrodes_number())  # to replace, use in built mea function
positions_y = np.zeros(bisc.get_electrodes_number())
positions_z = np.zeros(bisc.get_electrodes_number())

for i in range(bisc.get_electrodes_number()):  # to replace
    positions_x[i] = bisc.positions[i][0]
    positions_y[i] = bisc.positions[i][1]
    positions_z[i] = bisc.positions[i][2]

print('simulating...')
ExtPot = utils.ImposedPotentialField(source_amps, positions_x, positions_y, positions_z, sigma)  # to replace, use in built mea function
v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
v_cell_ext[:, :] = ExtPot.ext_field(cell.xmid, cell.ymid, cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

cell.insert_v_ext(v_cell_ext, t)
cell.simulate(rec_vmem=True, rec_imem=True)

os.chdir(CWD)


# FIGURES
n_sec, names = get_sections_number(cell)

fig = plt.figure(figsize=[20, 8])
fig.subplots_adjust(wspace=0.1)

ax1 = plt.subplot(131, projection="3d",
                  title="", aspect='auto', xlabel="x [$\mu$m]",
                  ylabel="y [$\mu$m]", zlabel="z [$\mu$m]")

mea.plot_probe_3d(bisc, ax=ax1, xlim=[-300, 300], ylim=[-300, 300], zlim=[np.min(cell.zmid) - 50, 200], type='planar')

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=-100, vmax=50)

colr = plt.cm.Set2(np.arange(n_sec))
for i, sec in enumerate(names):
    [ax1.plot([cell.xstart[idx], cell.xend[idx]],
              [cell.ystart[idx], cell.yend[idx]],
              [cell.zstart[idx], cell.zend[idx]],
              '-', c=colr[i], clip_on=False) for idx in cell.get_idx(sec)]
    if sec != 'soma':
        ax1.plot([cell.xstart[cell.get_idx(sec)[0]], cell.xend[cell.get_idx(sec)[0]]],
                 [cell.ystart[cell.get_idx(sec)[0]], cell.yend[cell.get_idx(sec)[0]]],
                 [cell.zstart[cell.get_idx(sec)[0]], cell.zend[cell.get_idx(sec)[0]]],
                 '-', c=colr[i], clip_on=False, label=sec)
ax1.scatter(cell.xmid[cell.get_idx('soma')[0]], cell.ymid[cell.get_idx('soma')[0]],
            cell.zmid[cell.get_idx('soma')[0]], s=33, marker='o', c='k', alpha=.7, label='soma')


n_sample_axon = 2  # number of points along the z axis from which data will be displayed in figures (+ soma)
n_sample_dend = 3
n_sample_apic = 3
n_sample = n_sample_axon + n_sample_dend + n_sample_apic

store_idx = []
color = iter(plt.cm.rainbow(np.linspace(0, 1, n_sample)))
for i in np.linspace(int(np.min(cell.zend)), int(np.max(cell.zend)), n_sample_axon, dtype=int):
    col = next(color)
    idx = cell.get_closest_idx(z=i, section='axon')
    ax1.scatter(cell.xmid[idx], cell.ymid[idx],
                cell.zmid[idx], s=50, c=col, marker='*', alpha=.5, label=cell.get_idx_name(idx)[1])
    store_idx.append(idx)

for i in random.sample(cell.get_idx('dend').tolist(), n_sample_dend):
    col = next(color)
    ax1.scatter(cell.xmid[i], cell.ymid[i],
                cell.zmid[i], s=50, c=col, marker='*', alpha=.5, label=cell.get_idx_name(i)[1])
    store_idx.append(i)

for i in random.sample(cell.get_idx('apic').tolist(), n_sample_apic):
    col = next(color)
    ax1.scatter(cell.xmid[i], cell.ymid[i],
                cell.zmid[i], s=50, c=col, marker='*', alpha=.5, label=cell.get_idx_name(i)[1])
    store_idx.append(i)

fig.tight_layout()

art = []
lgd = ax1.legend(loc=9, prop={'size': 12}, bbox_to_anchor=(1.5, .6), ncol=1)
art.append(lgd)


elev = 15     # Default 30
azim = 45    # Default 0
ax1.view_init(elev, azim)


ax2 = plt.subplot(133, title="Vmem", aspect='auto', xlabel="time [ms]",
                  ylabel="[mV]")
color = iter(plt.cm.rainbow(np.linspace(0, 1, n_sample)))
for idx in store_idx:
    ax2.plot(cell.tvec, cell.vmem[idx], c=next(color))
ax2.plot(cell.tvec, cell.vmem[0], c='k', label='soma')

# plt.savefig(os.path.join(output_f, NRN + '_' + name_shape_ecog + '_' + str(amp) + '.png'), dpi=300)

plt.figure(3)
plt.title('current pulse')
plt.plot(t, pulse)
plt.xlabel('[ms]')
plt.ylabel('[$\mu$A]')

plt.show()
