# Python LAMMPS Trajectory parser (Python 3.7 64-bit)
# Last modified 2021-07-19
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Establishing where the program is running from/which directory it was called from
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) # https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory helpful
cwd = os.getcwd()

# User-defined flags
numArgs = len(sys.argv) - 1
if numArgs < 3:
	raise AttributeError("""
	Missing arguments! Required arguments are:
	- parseDescendants ('true', otherwise only children will parsed, case is auto-adjusted to lower),
	- x1 (float, starting point of wall),
	- x2 (float, ending point of wall).""") # not sure which is the correct error type...
parseDescendants = False # (sys.argv[1].lower() == "true") # temporary override since I haven't gone into the complexity of recursive file searching
x1 = float(sys.argv[2])
x2 = float(sys.argv[3])

# Constants
MIN_NAME_LENGTH = len("x.lammpstrj")
TIMESTEP = 0.00005 # ns
RF = 8.6 # nm, flory radius (equilibrium radius of gyration for 64mer)

# Looping through each child directory and finding any file that a) ends in .lammpstrj (LAMMPS trajectory file) and b) is not a wall file.
# Note that we can choose to parse through descendants or only direct children - this is useful as a flag depending on how files are laid out.
fileList = []
if not parseDescendants:
	print("Identified files:")
	for name in os.listdir(cwd):
		# Filtering out files we don't want to read
		if not os.path.isfile(os.path.join(cwd, name)): # we can use this also for recursive descent
			continue
		if len(name) < MIN_NAME_LENGTH: # need to make sure we don't index out of bounds
			continue
		fileName = name[(len(name)-MIN_NAME_LENGTH+1):(len(name))] # grabbing the end of the file name using the colon operator
		isWall = name[len(name)-MIN_NAME_LENGTH] == "w" # checking if the file is a wall file; we don't want to parse wall files (yet)
		if fileName != ".lammpstrj" or isWall:
			continue
		
		# Files that made it out of the filter process
		fileDirectory = os.path.join(cwd, name)
		print(fileDirectory)
		fileList.append(fileDirectory)

else:
	print("hi")
	
# Reading data from the dump files
# LAMMPS Trajectory files have 9 lines of dump info (for the dumps we are using; this can be different depending on the user configuration).
# A more robust parser would be able to parse any type of dump file but for now this is adequate.

# 1 TIMESTEP
# 2 []
# 3 NUMBER OF ATOMS
# 4 []
# 5 BOX BOUNDS
# 6 [x1] [x2]
# 7 [y1] [y2]
# 8 [z1] [z2]
# 9 ATOM POSITION INFO
# _ [ATOM ID] [x] [y] [z] [vx] [vy] [vz]
# NOTE: ATOM ID is NOT ALWAYS IN THE SAME ORDER - always grab the ID wrt the data in each line.

# The last line is listed for each atom, dependant on the number of atoms provided on line 4.
# This will repeat for the remainder of the dump, so given the size of the file, we can loop through in chunks of 9+(#Atoms).
polymerList = []
polymerMeta = [] # meta data regarding each polymer: number of atoms, number of dumps, timestep, etc...
for dumpFile in fileList:
	# Initializing read
	file = open(dumpFile, 'r')
	lines = file.readlines()
	if len(lines) < 4: # can't check number of atoms
		continue
	numAtoms = int(lines[3][0:(len(lines[3])-1)]) # avoiding /n characters at the end of lines
	dumpLength = numAtoms + 9
	numDumps = len(lines)/dumpLength
	if len(lines) < dumpLength + 2: # can't check dump time
		continue 
	dumpStep = int(lines[dumpLength + 1][0:(len(lines[dumpLength + 1])-1)])
	newPolymerData = (np.empty((numAtoms, int(numDumps)), dtype=float))

	# Identifying atoms in order (atom ID needs to be preprocessed for arrays)
	atomList = []
	for i in range(0, numAtoms):
		lineIndex = i + 9
		currentLine = lines[lineIndex]
		lineElements = currentLine.split()
		atomId = int(lineElements[0])
		atomList.append(atomId)
	atomList = sorted(atomList)
	idToIndex = {} # a lookup table to convert atomId values to array values based on the order of the polymer
	for i in range(0, numAtoms):
		idToIndex[atomList[i]] = i

	# Recording data
	for i in range(0, int(numDumps)):
		dumpIndex = i*dumpLength + 9
		for j in range(dumpIndex, dumpIndex + numAtoms):
			currentLine = lines[j]
			lineElements = currentLine.split()
			atomId = int(lineElements[0])
			xPos = float(lineElements[1])
			atomIndex = idToIndex[atomId]
			newPolymerData[atomIndex][i] = xPos

	polymerList.append(newPolymerData)
	polymerMeta.append({
		"numAtoms": numAtoms,
		"numDumps": numDumps,
		"dumpStep": dumpStep,
	})
	file.close() # make sure to close the file, don't need any funny things happening to it

# Pre-processing before graphing
firstMonomerIndex = []
firstMonomerTime = []
distAxis = []
timeAxis = []
for i in range(0, len(polymerList)): # I'm using this funny loop because I need the index 'i', not really python style but it works
	polymerData = polymerList[i]
	distArray = [
		np.empty((int(polymerMeta[i]["numDumps"]), ), dtype=float), # 3 individual (same size) arrays holding data for 3 key monomers,
		np.empty((int(polymerMeta[i]["numDumps"]), ), dtype=float), #		the head, tail, and leading monomers.
		np.empty((int(polymerMeta[i]["numDumps"]), ), dtype=float)
	]
	timeArray = [
		np.empty((int(polymerMeta[i]["numDumps"]), ), dtype=float) # only need one here; time coords are the same for all monomers
	]

	# Find first monomer to reach entrance of wall
	firstMonomerFound = False # we need this flag to know when to stop checking for the first monomer
	firstMonomerCoords = [] # technically not required but I prefer to pre-declare variables that are used outside of loops
	for t in range(0, int(polymerMeta[i]["numDumps"])): # loop in time first, so we check each monomer before the next timestep
		for j in range(0, polymerMeta[i]["numAtoms"]):
			xPos = polymerData[j][t]
			if xPos >= x1 and not firstMonomerFound: # here we can also check against the middle of the wall, end of the wall, average time, etc...
				firstMonomerFound = True
				firstMonomerCoords = [j, t]
			polymerData[j][t] /= RF # normalizing for graphing, making sure we do this after the comparison above which is not in normalized units
	firstMonomerIndex.append(firstMonomerCoords[0])
	firstMonomerTime.append(firstMonomerCoords[1])
	print(i, firstMonomerIndex[i], firstMonomerTime[i])

	# Setting up arrays for graphing
	for t in range(0, int(polymerMeta[i]["numDumps"])):
		distArray[0][t] = polymerData[0][t]
		distArray[1][t] = polymerData[polymerMeta[i]["numAtoms"]-1][t]
		distArray[2][t] = polymerData[firstMonomerIndex[i]][t]
		timeArray[0][t] = (t - firstMonomerTime[i])*TIMESTEP*polymerMeta[i]["dumpStep"] # changing to correct units (ns)

	distAxis.append(distArray)
	timeAxis.append(timeArray)

# Graphing
fig1 = plt.figure()
for i in range(0, len(polymerList)): # prob required to average instead of plot a bunch
	# main plot
	currentTimeAxis = timeAxis[i]
	currentDistAxis = distAxis[i] # note there are 3 dist axes, so we need 3 separate plotters (will plot on same graph but each one is grouped)
	plt.plot(
		currentTimeAxis[0], currentDistAxis[0], '--r', 
		marker='o', markevery=300, markerfacecolor='none', markeredgecolor=(0.3, 0, 0), markeredgewidth=2,
		label='Head'
	)
	plt.plot(
		currentTimeAxis[0], currentDistAxis[1], '--g', 
		marker='^', markevery=300, markerfacecolor='none', markeredgecolor=(0, 0.3, 0), markeredgewidth=2,
		label='Tail'
	)
	plt.plot(
		currentTimeAxis[0], currentDistAxis[2], '--b', 
		marker='s', markevery=300, markerfacecolor='none', markeredgecolor=(0, 0, 0.3), markeredgewidth=2,
		label='Front'
	)

	# plot settings
	plt.xlabel('$t - t_a (ns)$')
	yAxisLabel = plt.ylabel(r'$\frac{\Delta r}{R_F}$')
	yAxisLabel.set_rotation(0)
	yAxisLabel.set_size(20)
	plt.legend()

plt.show()
plt.savefig("TEST PLOT")