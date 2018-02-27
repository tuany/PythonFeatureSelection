import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler

gaussiannb = {
				'Minimum Redundace Maximum Relevance':[(6,0.7641283525),(8,0.747217661),(13,0.7373198321),(17,0.7459177157),(22,0.7644362343),(114,0.7548690932),(227,0.745609834)],
				'Fast Correlation Filter':[(6,0.7200784528),(8,0.7379584018),(13,0.7641283525),(17,0.7637976647),(17,0.7548690932),(22,0.7548690932),(114,0.745609834),(227,0.745609834)],
				'Correlation Feature Selection':[(6,0.7564769203),(8,0.7644590403),(13,0.7644590403),(17,0.7373198321),(22,0.7465790914),(114,0.7637976647),(227,0.745609834)],
				'ReliefF':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'Principal Component Analysis':[(6,0.7130655902),(8,0.6862798759),(13,0.7293377121),(17,0.7481641124),(22,0.721355592),(114,0.6307015143),(227,0.6307015143)],
				'Robust Feature Selection':[(6,0.7740261814),(8,0.7555076628),(13,0.7644362343),(17,0.7644362343),(22,0.7736954935),(114,0.745609834),(227,0.745609834)]
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

# svm params: C = 5, gamma = 1, random_state = a essa hora nao sei
svmlin = {
			'Minimum Redundace Maximum Relevance':[(6,0.6677385514),(8,0.6766671228),(13,0.6677385514),(17,0.685926382),(22,0.685926382),(114,0.7650748039),(227,0.7392127349)],
			'Fast Correlation Filter':[(6,0.6674078635),(8,0.6670771757),(13,0.6677385514),(17,0.685926382),(22,0.685926382),(114,0.7564541142),(227,0.7392127349)],
			'Correlation Feature Selection':[(6,0.6307015143),(8,0.6492200328),(13,0.6670771757),(17,0.6951856413),(22,0.6584792921),(114,0.7564541142),(227,0.7302841635)],
			'ReliefF':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
			'Principal Component Analysis':[(6,0.7395434227),(8,0.739235541),(13,0.7762497719),(17,0.748802682),(22,0.7580619413),(114,0.7395434227),(227,0.7395434227)],
			'Robust Feature Selection':[(6,0.648889345),(8,0.6766671228),(13,0.648889345),(17,0.6769978106),(22,0.6856185003),(114,0.7216634738),(227,0.7392127349)]
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

# svm poly, C = 0.3 e gamma = 0.001, degree = 3 
svmpoly = {
				'Minimum Redundace Maximum Relevance':[(6,0.6307015143),(8,0.6396300858),(13,0.6674078635),(17,0.6766671228),(22,0.6674078635),(114,0.7564541142),(227,0.7305920452)],
				'Fast Correlation Filter':[(6,0.6396300858),(8,0.648889345),(13,0.6766671228),(17,0.6677385514),(22,0.6584792921),(227,0.7475255428),(114,0.7299762817)],
				'Correlation Feature Selection':[(6,0.6307015143),(8,0.6307015143),(13,0.6581486043),(17,0.6766671228),(22,0.6766671228),(114,0.7478334246),(227,0.7305920452)],
				'ReliefF':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
				'Principal Component Analysis':[(6,0.7481641124),(8,0.7395434227),(13,0.7669905127),(22,0.7669905127),(17,0.7484719942),(114,0.7484719942),(227,0.7484719942)],
				'Robust Feature Selection':[(6,0.6307015143),(8,0.648889345),(13,0.6855956942),(17,0.6766671228),(22,0.6492200328),(114,0.7478562306),(227,0.7392127349)]
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

# svm rbf, C = 100 e gamma = 0.01
svmrbf = {
			'Minimum Redundace Maximum Relevance':[(6,0.6307015143),(8,0.6307015143),(13,0.6581486043),(17,0.6674078635),(22,0.6588099799),(114,0.7389276592),(227,0.7475255428)],
			'Fast Correlation Filter':[(6,0.6307015143),(8,0.6307015143),(13,0.648889345),(17,0.6766671228),(22,0.676336435),(114,0.7299762817),(227,0.747194855)],
			'Correlation Feature Selection':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6492200328),(22,0.676336435),(114,0.7210477103),(227,0.7389048531)],
			'ReliefF':[(6,0.6307015143),(8,0.6307015143),(13,0.6307015143),(17,0.6307015143),(22,0.6307015143),(114,0.6307015143),(227,0.6307015143)],
			'Principal Component Analysis':[(6,0.7216634738),(8,0.7302841635),(13,0.748802682),(17,0.748802682),(22,0.7673212005),(114,0.748802682),(227,0.748802682)],
			'Robust Feature Selection':[(6,0.6307015143),(8,0.6307015143),(13,0.648889345),(17,0.6766671228),(22,0.6766671228),(114,0.7204091407),(227,0.7299762817)]
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

# svm sigmoid, C = 1 e gamma = 0.01 (default)
svmsig = {
				'Minimum Redundace Maximum Relevance':[(6,0.630),(8,0.630),(13,0.630),(17,0.630),(22,0.630),(114,0.630),(227,0.630)],
				'Fast Correlation Filter':[(6,0.631),(8,0.631),(13,0.631),(17,0.631),(22,0.631),(114,0.631),(227,0.631)],
				'Correlation Feature Selection':[(6,0.6),(8,0.6),(13,0.6),(17,0.6),(22,0.6),(114,0.6),(227,0.6)],
				'ReliefF':[(6,0.6396300858),(8,0.6830414158),(13,0.6214422551),(17,0.547368181),(22,0.6121829958),(114,0.6307015143),(227,0.6307015143)],
				'Principal Component Analysis':[(6,0.6581486043),(8,0.6581486043),(13,0.6581486043),(17,0.6581486043),(22,0.6581486043),(114,0.6581486043),(227,0.6581486043)],
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