import numpy as np
import cPickle as cp
from matplotlib import pyplot as pp
import time

class springworld:
	
	
	def __init__(self, init_pos, linsprings, init_vel=0.0, nd=2,
						masses=1.0, dt=0.002, G=8.0, lin_damping=10.0):
		
		#xpos, ypos: initial mass positions
		#masses: masses of each point
		#spring_consts: matrix of spring constants - triangular matrix
		#spring lengs: matrix of spring lengths - also triangular
		#dt: temporal stepsize
		
		self.nd=nd
		self.nm=len(init_pos)
		self.nls=len(linsprings)
		self.pos=np.asarray(init_pos,dtype='float32')
		if init_vel==0.0:
			self.vel=np.zeros((self.nm,nd))
		self.masses=np.ones(self.nm)*masses
		self.linsprings=np.asarray(linsprings)
		self.dt=dt
		self.G=G
		self.lin_damping=lin_damping
	
	
	def compute_forces(self, thrust=None):
		
		#this function computes the sum of all the forces on each point mass
		
		ftot=np.zeros((self.nm, self.nd))
		
		rot=np.asarray([[0,1],[-1,0]],dtype='float32')
		
		#linear spring and thrust forces
		for s in self.linsprings:
			k,l,i,j=s
			dvec=self.pos[i]-self.pos[j]
			dist=np.sqrt(np.sum(dvec**2))
			fs=-k*(dist-l)
			
			#damping
			fd=np.dot(dvec/dist, self.vel[i]-self.vel[j])*self.lin_damping
			
			#thrust
			if thrust!=None:
				ft=np.dot(rot,dvec/dist)*thrust[i]
			else:
				ft=0.0
			
			
			ff=(dvec/dist)*(fs-fd)
			
			ftot[i]+=ff; ftot[j]-=ff+ft
		
		#drag
		ftot-=0.1*self.vel	
		
		#gravity
		ftot[:,1]-=self.G
		
		
		
		return ftot
	
	
	def compute_accel(self,f):
		
		#for now assuming that all the masses are 1.0
		
		return f
	
	
	def sim_dynamics(self, nsteps, thrust=None):
		
		for i in range(nsteps):
			f = self.compute_forces(thrust)
			#a = self.compute_accel(f)
			self.vel+=f*self.dt
			self.pos+=self.vel*self.dt
			
			#first point is anchored to (0,0)
			self.pos[0]*=0.0; self.vel[0]*=0.0
	
	
	def get_pos(self):
		
		return self.pos
	
	
	def get_proprio(self):
		
		#this function returns two lists, one with the current lengths 
		#of each linear spring, and one with the current angles of each 
		#rotational spring
		
		lengths=[]
		contact=np.zeros(self.nm)
		for s in self.linsprings:
			k,l,i,j=s
			dvec=self.pos[i]-self.pos[j]
			dist=np.sqrt(np.sum(dvec**2))
			lengths.append(dist)
		
		
		for i in range(self.nm):
			if self.pos[i,1]<0.0:
				contact[i]=1.0
		
		return lengths, contact
	
	
	def change_params(self, newlin):
		for x in newlin:
			i,newval=x
			self.linsprings[i,1]=newval
	

def make_chain(nnodes,link,lengths):
	
	pos=np.zeros((nnodes*2,2))
	for i in range(nnodes):
		pos[2*i,0]=i*lengths; pos[2*i,1]=0.0
		pos[2*i+1,0]=i*lengths; pos[2*i+1,1]=lengths
	
	
	linsprings=[]
	#vertical springs
	for i in range(nnodes):
		linsprings.append([link, lengths, 2*i,2*i+1])
	
	#top horizontal and diagonal
	for i in range(nnodes-1):
		linsprings.append([link, lengths, 2*i+1,2*i+3])
		linsprings.append([link, np.sqrt(2.0)*lengths, 2*i+1, 2*i+2])
	
	#bottom horizontal and diagonal
	for i in range(nnodes-1):
		linsprings.append([link, lengths, 2*i,2*i+2])
		linsprings.append([link, np.sqrt(2.0)*lengths, 2*i, 2*i+3])
	
	
	return pos, linsprings

def make_light_chain(nnodes, link, lengths):
	
	pos=np.zeros((nnodes*2-1,2))
	for i in range(len(pos)):
		pos[i,0]=0.5*lengths*float(i)
		if i%2!=0:
			pos[i,1]=lengths*np.sqrt(3.0/4.0)
	
	linsprings=[]
	for i in range(nnodes-1):
		linsprings.append([link, lengths, 2*i, 2*(i+1)])
	
	for i in range(nnodes-2):
		linsprings.append([link, lengths, 2*i+1, 2*(i+1)+1])
	
	for i in range((nnodes-1)*2):
		linsprings.append([link, lengths, i, i+1])
		
	return pos, linsprings


def animate_hist(pos):
	
	x=pos[:,:,0]; y=pos[:,:,1]
	nt=len(x)
	pp.ion()
	fig=pp.figure()
	ax1=fig.add_subplot(111)
	ax1.set_xlim((-12.0,20.0))
	ax1.set_ylim((-12.0,12.0))
	line1, =ax1.plot(x[0],y[0],"o",markersize=12)
	
	for i in range(nt-1):
		line1.set_xdata(x[i+1])
		line1.set_ydata(y[i+1])
		fig.canvas.draw()
		time.sleep(0.01)


