import numpy as np
import cPickle as cp
from matplotlib import pyplot as pp

import double_pendulum_world
from double_pendulum_world import springworld as world

def pink_noise(nsamps,ampbase,cutoff):
	out=np.zeros(nsamps,dtype='float32')
	x=np.arange(nsamps)
	nsamps=float(nsamps)
	for i in np.arange(np.floor(nsamps/2)):
		phase=2.0*np.pi*np.random.uniform()
		amp=np.exp(-(i/cutoff)**2)*np.random.normal(1)
		y=amp*np.sin(x*i*2.0*np.pi/nsamps+phase)
		out=out+y
	return out*ampbase


nm=3
nd=2
nt=2000

#init_pos, linsprings = springworld_old_linonly.make_light_chain(nm,500.0,2.0)

#init_pos[:,1]+=0.01

init_pos=np.asarray([[0,0],[0,4],[.02,8]],dtype='float32')

linsprings=[[2000.0,4.0,0,1],[2000.0,4.0,1,2]]

model=world(init_pos,linsprings)

ns=4*(nm-2)+3+2*nm-1

a=np.zeros((nt,7))



#thrust=np.arange(nt)/20.0
#thrust=np.asarray([np.sin(thrust),np.cos(thrust)])
#thrust=thrust.T

thrust=np.ones((2,nt))*0
thrust[0,:]=0.0
thrust=thrust.T

ph=np.zeros((nt,nm,nd))
vh=np.zeros((nt,nm,nd))

sh=np.zeros((nt,ns))

for i in range(nt):
	model.sim_dynamics(40,thrust[i])
	#lens, contact = model.get_proprio()
	#s=np.concatenate([lens,contact])
	#sh[i]=s
	#model.change_params([[0, a[i,0]],[1, a[i,1]], [2, a[i,2]], [3, a[i,3]], [4, a[i,4]], [5, a[i,5]], [6, a[i,6]]])
	
	ph[i]=model.pos
	vh[i]=model.vel

pp.plot(ph[:,:,0])
pp.show()

double_pendulum_world.animate_hist(ph)

f=open('tdata.cpl','wb')
cp.dump(ph,f,2)
f.close()
