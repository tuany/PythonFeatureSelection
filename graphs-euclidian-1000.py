import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler

gaussiannb = {
				'Minimum Redundace Maximum Relevance':[(6,0.7644362343),(8,0.7462484036),(13,0.755176975),(17,0.7462484036),(22,0.7555076628),(114,0.7363505747),(227,0.745609834)],
				'Fast Correlation Filter':[(6,0.7555076628),(8,0.7376277139),(13,0.7465562853),(17,0.7372970261),(22,0.7369891443),(114,0.7548690932),(227,0.745609834)],
				'Correlation Feature Selection':[(6,0.7280605729),(8,0.7283684547),(13,0.7286991425),(17,0.7188013136),(22,0.7462484036),(114,0.7548690932),(227,0.745609834)],
				'ReliefF':[(6,0.3692984857),(8,0.3692984857),(13,0.3692984857),(17,0.3692984857),(22,0.3692984857),(114,0.3692984857),(227,0.3692984857)],
				'Principal Component Analysis':[(6,0.7130655902),(8,0.6862798759),(13,0.7293377121),(17,0.7481641124),(22,0.721355592),(114,0.6307015143),(227,0.6307015143)],
				'Robust Feature Selection':[(6,0.7184706258),(8,0.7363505747),(13,0.7363505747),(17,0.7452791461),(22,0.7545384054),(114,0.7548690932),(227,0.7545384054)]
			 }

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm', 'c', 'y'])))

fig = plt.figure()
ax = fig.add_subplot(111)
xlabels = [6, 114, 227]
plt.xticks(xlabels)
plt.xticks(rotation=45)
axes = plt.gca()
axes.set_xlim([5,228])
axes.set_ylim([0.0,0.8])

for k,v in gaussiannb.iteritems():
	plt.plot(*zip(*gaussiannb[k]), label=k, marker='.', linestyle='--')

# plt.grid()
plt.xlabel(r"""$n\'um. caracter\'isticas$""")
plt.ylabel(r"""$acur\'acia$""")

# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.4), shadow=True, ncol=1)
plt.legend()
plt.show()

# svm params: C = 1, gamma = 0.01, random_state = a essa hora nao sei
svmlin = {
			'Minimum Redundace Maximum Relevance':[(6,0.7219713556),(8,0.7283684547),(13,0.693885696),(17,0.7037835249),(22,0.6574872286),(114,0.576445904), (227,0.5943030469)],
			'Fast Correlation Filter':[(6,0.7207170224),(8,0.7025291918), (13,0.6313400839),(17,0.747194855), (22,0.7223020434),(114,0.6217729429),(227,0.6128215654)],
			'Correlation Feature Selection':[(6,0.6773056924),(8,0.7124042146),(13,0.711765645),(17,0.6753443715),(22,0.7130427842),(114,0.5493066959),(227,0.5767537858)],
			'ReliefF':[(6,0.3692984857),(8,0.3692984857),(13,0.3692984857),(17,0.3692984857),(22,0.3692984857),(114,0.3692984857),(227,0.3692984857)],
			'Principal Component Analysis':[(6,0.7663519431),(8,0.7216862799),(13,0.7494412516),(17,0.693885696),(22,0.6753671775),(114,0.6038701879),(227,0.6038701879)],
			'Robust Feature Selection':[(6,0.6578179164),(8,0.6670771757),(13,0.6769750046),(17,0.6951400292),(22,0.7133506659),(114,0.6032544244),(227,0.5946337347)]
		 }

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm', 'c', 'y'])))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(xlabels)
plt.xticks(rotation=45)
axes = plt.gca()
axes.set_xlim([5,228])
axes.set_ylim([0.0,0.8])

for k in svmlin.keys():
	print(svmlin[k])
	plt.plot(*zip(*svmlin[k]), label=k, marker='.', linestyle='--')

# plt.grid()
plt.xlabel(r"""$n\'um. caracter\'isticas$""")
plt.ylabel(r"""$acur\'acia$""")

# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.4), shadow=True, ncol=1)
plt.legend()
plt.show()

# svm poly, C = 0.3 e gamma = 1, degree = 2 
svmpoly = {
				'Minimum Redundace Maximum Relevance':[(6,0.7207398285),(8,0.729314906),(13,0.6300629447),(17,0.6674078635),(22,0.6575100347),(114,0.5847359059),(227,0.6124908776)],
				'Fast Correlation Filter':[(6,0.711788451),(8,0.6929620507),(13,0.6846492428),(17,0.7289842182),(22,0.6207808794),(114,0.5949644226),(227,0.6032316183)],
				'Correlation Feature Selection':[(6,0.6862570699),(8,0.7216634738),(13,0.7028370735),(17,0.7210249042),(22,0.6210887612),(114,0.6313400839),(227,0.5952723043)],
				'ReliefF':[(6,0.4341133005),(8,0.4433725598),(13,0.4711503375),(17,0.3878170042),(22,0.3692984857),(114,0.3692984857),(227,0.3692984857)],
				'Principal Component Analysis':[(6,0.7315613027),(8,0.6664614122),(13,0.6380678708),(17,0.6469736362),(22,0.6482507754),(114,0.6753899836),(227,0.6753899836)],
				'Robust Feature Selection':[(6,0.6929392447),(8,0.6756978654),(13,0.7302841635),(17,0.685903576),(22,0.6941935778),(114,0.6137908228),(227,0.6131522532)]
			 }

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm', 'c', 'y'])))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(xlabels)
plt.xticks(rotation=45)
axes = plt.gca()
axes.set_xlim([5,228])
axes.set_ylim([0.0,0.8])

for k,v in svmpoly.iteritems():
	plt.plot(*zip(*svmpoly[k]), label=k, marker='.', linestyle='--')

# plt.grid()
plt.xlabel(r"""$n\'um. caracter\'isticas$""")
plt.ylabel(r"""$acur\'acia$""")

# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.4), shadow=True, ncol=1)
plt.legend()
plt.show()

# svm rbf, C = 0.7 e gamma = 0.0001
svmrbf = {
				'Minimum Redundace Maximum Relevance':[(6,0.6852878124),(8,0.7296683999),(13,0.7025291918),(17,0.7025291918),(22,0.7197705711),(114,0.6568714651),(227,0.6307015143)],
				'Fast Correlation Filter':[(6,0.7037835249),(8,0.7111498814),(13,0.6852878124),(17,0.7551997811),(22,0.7558383507),(114,0.674751414),(227,0.6307015143)],
				'Correlation Feature Selection':[(6,0.7031449553),(8,0.7283684547),(13,0.7283684547),(17,0.747217661),(22,0.7561462324),(114,0.6568714651),(227,0.6307015143)],
				'ReliefF':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'Principal Component Analysis':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'Robust Feature Selection':[(6,0.6932699325),(8,0.7021985039),(13,0.7200784528),(17,0.7469097792),(22,0.7373198321),(114,0.6479200876),(227,0.6307015143)]
			 }

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm', 'c', 'y'])))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(xlabels)
plt.xticks(rotation=45)
axes = plt.gca()
axes.set_xlim([5,228])
axes.set_ylim([0.0,0.8])

for k,v in svmrbf.iteritems():
	plt.plot(*zip(*svmrbf[k]), label=k, marker='.', linestyle='--')

# plt.grid()
plt.xlabel(r"""$n\'um. caracter\'isticas$""")
plt.ylabel(r"""$acur\'acia$""")

# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.4), shadow=True, ncol=1)
plt.legend()
plt.show()

# svm sigmoid, C = 0.3 e gamma = 0.0004 (default)
svmsig = {
				'Minimum Redundace Maximum Relevance':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'Fast Correlation Filter':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'Correlation Feature Selection':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'ReliefF':[(6,0.6396300858),(8,0.6830414158),(13,0.6214422551),(17,0.547368181),(22,0.6121829958),(114,0.6307015143),(227,0.6307015143)],
				'Principal Component Analysis':[(6,0.7369663383),(8,0.7462255975),(13,0.7554848568),(17,0.7462255975),(22,0.7462255975),(114,0.7462255975),(227,0.7462255975)],
				'Robust Feature Selection':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)]
			 }

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm', 'c', 'y'])))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(xlabels)
plt.xticks(rotation=45)
axes = plt.gca()
axes.set_xlim([5,228])
axes.set_ylim([0.0,0.8])

for k,v in svmsig.iteritems():
	plt.plot(*zip(*svmsig[k]), label=k, marker='.', linestyle='--')

# plt.grid()
plt.xlabel(r"""$n\'um. caracter\'isticas$""")
plt.ylabel(r"""$acur\'acia$""")

# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.4), shadow=True, ncol=1)
plt.legend()
plt.show()