

"""
core.py: Simulation of Thomas Point Process

"""
__author__ = "Subharthi Banerjee"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Subharthi Banerjee"
__email__ = "sbanerjee15@huskers.unl.edu"
__status__ = "Development"


## 
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import threading
from time import time
import concurrent.futures
import matplotlib.animation as animation



# have an import for gps coordinate

class _PoissonPP():
	"""
	Poisson Point Process (used internally)
	Attributes:


	Methods/Properties:

	Dunders:


	"""
	def __init__(self, lam=0.1, nUE=100, seed=2019):
		"""
		Constructor

		init_args:



		"""
		self._lam = lam
		self._nUE = nUE
		self.seed = seed
		self._dim = 2
		self._nParents = None
		self._locParents = None
		np.random.seed(seed)

	def __repr__(self):
		return "Poisson: lambda = {0}, nUE = {1}, seed = {2}, nParents = {3}".format(self.get_lam(), \
			self.get_nUE(), self.seed, self.get_nParents())

	def get_lam(self):
		return self._lam
	def get_nUE(self):
		return self._nUE
	def get_dim(self):
		return self._dim
	def set_lam(self, lam):
		self._lam = lam
	def set_nUE(self, nUE):
		if isinstance(nUE, int) is not True:
			raise TypeError('Value should not be float')
		elif(nUE <= 0):
			raise ValueError('number of UE should be greater than 0')
		self._nUE = nUE
	def set_nParents(self, nParents):

		if isinstance(nParents, int) is not True:
			raise TypeError('Value should not be float')
		elif(nParents <= 0):
			raise ValueError('number of UE should be greater than 0')
		self._nParents = nParents
	def get_nParents(self):
		return self._nParents

	def set_locParents(self, locParents):
		self._locParents = locParents

	def get_locParents(self):
		#return self.get_nUE()*self._locParents
		return self._locParents
	def make_PP(self):
		self.set_nParents(np.random.poisson \
			(self.get_lam() * self.get_nUE()))

		self.set_locParents(np.random.uniform(0, 1, (self.get_nParents(), \
		 self.get_dim())))


class ThomasPP():
	"""
	Thomas Point Process (should be used repeatedly during different time instant)
	
	Attributes:


	Methods/Properties:

	Dunders:


	"""
	def __init__(self, kappa = 0.1, sigma = 0.01, mu = 20, Dx = 100, seed = 2019):
		"""

		"""
		self._kappa = kappa
		self._sigma = sigma
		self._mu = mu
		self._Dx = Dx
		self._seed = seed
		self._nParents = None
		self._locParents = None
		self._nChildren = None
		self.locChildrenx = None
		self.locChildreny = None
		self.locChildren = None
		self.parent_child_dict = {}
		self.sumChildren = None

	def set_all(self, kappa = 0.1, sigma = 0.01, mu = 20, Dx = 100, seed = 2019):
		self.set_kappa(kappa)
		self.set_sigma(sigma)
		self.set_mu(mu)
		self.set_Dx(Dx)
		self.set_seed(seed)

	def _init_locChildren(self):
		self.locChildren = np.zeros([sum(self.get_nChildren()), 2])
	def set_nParents(self, nParents):
		self._nParents = nParents

	def set_nChidren(self, nChildren):
		self._nChildren = nChildren

	def get_nParents(self):
		return self._nParents

	def get_locParents(self):
		return self._locParents
	def get_nChildren(self):
		return self._nChildren
	def set_nChildren(self, nChildren):
		self._nChildren = nChildren

	def set_locParents(self, locParents):
		self._locParents = locParents

	def set_nParents_locParents(self):

		poissonPP = _PoissonPP(self.get_kappa(), self.get_Dx())
		poissonPP.make_PP()
		print(poissonPP)
		self.set_nParents(poissonPP.get_nParents())
		self.set_locParents(poissonPP.get_locParents())

	def set_locChildren(self, locChildrenx, locChildreny):

		self.locChildren = np.zeros([sum(self.get_nChildren()), 2])
		self.locChildrenx = locChildrenx
		self.locChildreny = locChildreny
		self.locChildren[:, 0] = locChildrenx
		self.locChildren[:, 1] = locChildreny


	def get_kappa(self):
		"""
		"""
		return self._kappa

	def get_sigma(self):
		return self._sigma

	def get_mu(self):
		return self._mu

	def get_Dx(self):
		return self._Dx

	def get_seed(self):
		return self._seed

	def set_kappa(self, kappa):

		self._kappa = _kappa
	def set_sigma(self, sigma):

		self._sigma = sigma

	def set_mu(self, mu):
		self._mu = mu

	def set_Dx(self, Dx):
		self._Dx = Dx

	def set_seed(self, seed):
		self._seed = seed

	def __repr__(self):
		return 'TPP configured as: kappa = {0}, sigma = {1}, \
		mu = {2}, Dx = {3}, nParents = {4}, nChildren = {5}'.format(self.get_kappa(), \
			self.get_sigma(), self.get_mu(), self.get_Dx(), self.get_nParents(),\
			 self.get_nChildren())

	def time_it(func):

		"""
		A generic timer function
		"""
		def f(*args, **kwargs):
			start_time = time()
			val = func(*args, **kwargs)
			end_time = time()

			logging.info("time elapased %fs for function %s", end_time - start_time, func)
			return val
		return f

	@time_it
	def make_TPP_thread(self):


		#self.set_nParents_locParents()
		

		#logging.info("set number of parents: %d for the run", self.get_nParents())
		#nChildren = np.random.poisson(self.get_mu(), self.get_nParents())
		#self.set_nChildren(nChildren)
		#self._init_locChildren()
		with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
			executor.map(self.find_children_location_tf, range(2))



	

	
	def find_children_location_tf(self, index):

		"""
		"""

		sumChildren = sum(self.get_nChildren())

		logging.info('Inside thread %d with %d children', 
			index, sumChildren)
		nChildren = self.get_nChildren()


		
		
		children_coord = np.random.normal(0, self.get_sigma(), sumChildren)
		parent_coord = self.get_locParents()
		parent_coord_ind = parent_coord[:, index]


		#logging.info("Parent coorninate set in Thread %d", index)

		coord0 = np.repeat(parent_coord_ind, self.get_nChildren())

		children_coord = coord0 + children_coord 



		self.locChildren[:, index] = children_coord
		


		

	@time_it
	def make_TPP(self):
		"""
		"""


		locParents = self.get_locParents()
		locParentx = locParents[:, 0]


		locParenty = locParents[:, 1]

		#logging.info("total number of children %d", sumChildren, "with groups: ", nChildren)
		locChildrenx = np.random.normal(0, self.get_sigma(), self.sumChildren)
		locChildreny = np.random.normal(0, self.get_sigma(), self.sumChildren)

		
		
		
		# some temporary variable 
		xx0 = np.repeat(locParentx, self.get_nChildren())
		yy0 = np.repeat(locParenty, self.get_nChildren())

		#id0 = np.repeat(self.get_nParents(), self.get_nChildren())

		locChildrenx = xx0 + locChildrenx
		locChildreny = yy0 + locChildreny

		self.set_locChildren(locChildrenx, locChildreny)



	def init(self):
		self.set_nParents_locParents()
		

		logging.info("Initializing number of parents: %d for the run", self.get_nParents())
		nChildren = np.random.poisson(self.get_mu(), self.get_nParents())
		self.set_nChildren(nChildren)
		self._init_locChildren()
		

		sumChildren = sum(nChildren)
		self.sumChildren = sumChildren

	def start(self):
		"""
			run threaded or non threaded model
		"""

		if self._Dx * self._mu < 100_000:

			self.make_TPP()
		else:
			self.make_TPP_thread()

	def run(self):
		"""
		"""
		np.random.seed()
		self.start()
		return self.locChildren



class UpdateTPP():
	"""
	"""
	def __init__(self, tpp):
		self.tpp = tpp
		self.fig, self.ax = plt.subplots()
		self.scat = self.ax.scatter([], [], edgecolor='b', \
	facecolor='b', alpha=0.5)
		

		self.ax.set_xlim(0, self.tpp.get_Dx())
		self.ax.set_ylim(0, self.tpp.get_Dx())
		self.ax.grid(True)
		self.ani = animation.FuncAnimation(self.fig, self, interval=100, 
                                          init_func=self.init, blit=True)

	def __call__(self, i):

		self.tpp.run()

		self.scat.set_offsets(self.tpp.get_Dx()*self.tpp.locChildren[:, :2])
		logging.info("Logging at %d", i)
		return self.scat,


	def init(self):
		#self.ax.set_ylim(-1.1, 1.1)
	    #self.ax.set_xlim(0, 10)
	    #del xdata[:]
	    #del ydata[:]
	    self.scat.set_offsets(self.tpp.get_Dx()*self.tpp.locChildren[:, :2])
	    logging.info("Initiated plot")
	    return self.scat,




if __name__ == '__main__':

	os.system('clear')
	logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
	kappa = 0.1
	sigma = 0.01
	mu = 20
	Dx = 10_00
	seed = 2019

	tpp = ThomasPP(kappa, sigma, mu, Dx, seed)
	#tpp.set_all(kappa, sigma, mu, Dx, seed)

	tpp.init()
	tpp.start()

	print(tpp)

	
	
	#plt.scatter(tpp.get_Dx()*tpp.locChildren[:, 0],tpp.get_Dx()*tpp.locChildren[:, 1], edgecolor='b', \
	#facecolor='b', alpha=0.5)
	#plt.xlabel("x"); plt.ylabel("y");
	#plt.axis('equal');

	logging.info("Animation started")
	up = UpdateTPP(tpp)
	plt.show()
	logging.info("Animation finished")
	#plt.plot(tpp._locParents[:, 0], tpp._locParents[:, 1])

	
