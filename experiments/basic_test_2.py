import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T

import sys
sys.path.insert(0, '../SLmodel')
sys.path.insert(0, '../simulators')

from SLmodel_mean_h import SLmodel

import double_pendulum_world
from double_pendulum_world import springworld as sim

import math


def pink_noise(nsamps,ampbase,cutoff):
	out=np.zeros(nsamps,dtype='float32')
	x=np.arange(nsamps)/1000.0
	nsamps=float(nsamps)
	for i in np.arange(cutoff):
		phase=2.0*np.pi*np.random.uniform()
		amp=np.exp(-(i/cutoff)**2)*np.random.normal(1)
		y=amp*np.sin(x*i*2.0*np.pi+phase)
		out=out+y
	return out*ampbase


nx=16
ns=2
nh=2
npcl=20

nsamps=10
lrate=2e-5

dt=0.05
nt=300000

npred=1000

#making some data

x_hist=[]

theta=0.2
vec=np.ones(2)
M1=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype='float32')
M2=np.asarray([[np.cos(theta/3.0),-np.sin(theta/3.0)],[np.sin(theta/3.0),np.cos(theta/3.0)]],dtype='float32')
W=np.asarray(np.random.randn(nx,2),dtype='float32')
c=np.asarray(np.random.randn(nx)*1.0,dtype='float32')

v_hist=[]
v_hist.append(np.asarray([1.0,0.0]))


for i in range(nt):
	
	vec=v_hist[i]
	x_hist.append(np.dot(W,vec)+c+np.random.randn(nx)/100.0)
	if vec[0]>0.0:
		M=M1
	else:
		M=M2
	v_hist.append(np.dot(M,vec) + np.random.randn(2)/100.0)

vec=v_hist[-1]
xact=[]
for i in range(npred):
	
	xact.append(np.dot(W,vec)+c+np.random.randn(nx)/100.0)
	if vec[0]>0.0:
		M=M1
	else:
		M=M2
	vec=np.dot(M,vec) + np.random.randn(2)/100.0

xact=np.asarray(xact)

x_hist=np.asarray(x_hist,dtype='float32')

pp.plot(x_hist)
pp.show()

xdata=theano.shared(x_hist)

xvar=0.1

model=SLmodel(nx, ns, nh, npcl, xvar=xvar)

idx=T.lscalar()
x1=T.fvector()
x2=T.fvector()

#norm, eng, ssmp, sprd, Wx, updates0=model.forward_filter_step(x)
#norm, eng, updates0=model.forward_filter_step(x)
#inference_step=theano.function([x],[norm,eng,ssmp,sprd,Wx],updates=updates0,allow_input_downcast=True)
#inference_step=theano.function([x],[norm,eng],updates=updates0,allow_input_downcast=True)

#hsmps, updates0=model.forward_filter_step(x1)
#inference_step=theano.function([idx],hsmps,
								#updates=updates0,
								#givens={x1: xdata[idx,:]},
								#allow_input_downcast=True)
								
updates0=model.forward_filter_step(x1)
inference_step=theano.function([idx],None,
								updates=updates0,
								givens={x1: xdata[idx,:]},
								allow_input_downcast=True)

ess=model.get_ESS()
get_ESS=theano.function([],ess)

updates1=model.resample()
resample=theano.function([],updates=updates1)

lr=T.fscalar(); nsmps=T.lscalar()

nrg, updates2 = model.update_params(x1, x2, nsmps, lr)
learn_step=theano.function([idx,nsmps,lr],[nrg],
							updates=updates2,
							givens={x1: xdata[idx-1,:], x2: xdata[idx,:]},
							allow_input_downcast=True)

nps=T.lscalar()
sps, xps, hs, updates3 = model.simulate_forward(nps)
predict=theano.function([nps],[sps,xps,hs],updates=updates3,allow_input_downcast=True)


plr=T.fscalar()
pnsteps=T.lscalar()
ploss, updates4 = model.update_proposal_distrib(pnsteps,plr)
update_prop=theano.function([pnsteps, plr],ploss,updates=updates4,
							allow_input_downcast=True,
							on_unused_input='ignore')



h_hist=[]

e_hist=[]
ess_hist=[]
s_hist=[]
r_hist=[]
l_hist=[]
w_hist=[]
ploss_hist=[0.0]

th=[]

resample_counter=0
learn_counter=0

for i in range(nt-1):
	
	#normalizer,energies,ssamps,spreds,WTx=inference_step(vec)
	#normalizer,energies=inference_step(x)
	#h_samps=inference_step(i)
	inference_step(i)
	
	#pp.scatter(ssamps[:,0],ssamps[:,1],color='b')
	#pp.scatter(spreds[:,0],spreds[:,1],color='r')
	#pp.scatter(WTx[0],WTx[1],color='g')
	
	#pp.hist(energies,20)
	#pp.show()
	
	#print h_samps
	
	
	ESS=get_ESS()
	ess_hist.append(ESS)
	learn_counter+=1
	resample_counter+=1
	
	if resample_counter>0 and learn_counter>40:
		energy=learn_step(i,nsamps, lrate)
		e_hist.append(energy)
		learn_counter=0
		pl=update_prop(15,1e-3)
		ploss_hist.append(pl)
		l_hist.append(1)
		lrate=lrate*0.9997
	else:
		l_hist.append(0)
	
	if i%1000==0:	
		#print normalizer
		
		print 'Iteration ', i, ' ================================'
		print 'ESS: ', ESS
		print '\nParameters'
		print 'M'
		print model.M.get_value()
		print 'W'
		print model.W.get_value()
		print 'A'
		print model.A.get_value()
		print 'ph'
		print model.ph.get_value()
		print 'c'
		print model.c.get_value()
		print 'b'
		print model.b.get_value()
		print '\nMetaparameters:'
		print 'Proposal loss: ', ploss_hist[-1]
		print 'CCT-dot-true inverse covariance'
		W=model.W.get_value()
		b=model.b.get_value()
		C=model.C.get_value()
		cov_inv=np.dot(np.dot(C,C.T), np.dot(W.T, W)/(xvar**2)+np.eye(ns)*np.exp(-b))
		print cov_inv
		
	if ESS<npcl/2:
		resample()
		resample_counter=0
		r_hist.append(1)
	else:
		r_hist.append(0)
	
	s_hist.append(model.s_now.get_value())
	w_hist.append(model.weights_now.get_value())
	h_hist.append(model.h_now.get_value())
	
	if math.isnan(ESS):
		print '\nSAMPLING ERROR===================\n'
		print 'Proposal loss: ', ploss_hist[-1]
		print 'CCT-dot-true inverse covariance'
		W=model.W.get_value()
		b=model.b.get_value()
		C=model.C.get_value()
		cov_inv=np.dot(np.dot(C,C.T), np.dot(W.T, W)/(xvar**2)+np.eye(ns)*np.exp(-b))
		print cov_inv
		break




spred, xpred, hsmps = predict(npred)
#print spred.shape
#print spred
#print xpred.shape
#print xpred
#print hsmps.shape
#print hsmps

s_hist=np.asarray(s_hist)
w_hist=np.asarray(w_hist)
h_hist=np.asarray(h_hist)
ess_hist=np.asarray(ess_hist)
s_av=np.mean(s_hist,axis=1)



h_hist=h_hist*np.asarray(range(nh))
h_av=np.mean(h_hist,axis=2)
h_av=np.mean(h_hist,axis=1)



x_hist=np.asarray(x_hist)

e_hist=np.asarray(e_hist)
l_hist=np.asarray(l_hist)-.5
r_hist=np.asarray(r_hist)-.5
pp.figure(2)
pp.plot(x_hist)

pp.figure(3)
pp.plot(e_hist)
pp.figure(4)
pp.plot(s_av)
#pp.plot(r_hist, 'r')
#pp.plot(l_hist, 'k')

pp.figure(5)
pp.plot(ess_hist)

#pp.figure(5)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,0],color=zip(w_hist[:,i],np.zeros(len(w_hist)),1.0-w_hist[:,i]))

#pp.figure(6)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,1],color=zip(w_hist[:,i],np.zeros(len(w_hist)),1.0-w_hist[:,i]))
	
#pp.figure(7)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,0],color=zip(np.ones(len(w_hist)),np.zeros(len(w_hist)),np.zeros(len(w_hist)),w_hist[:,i]))



pp.figure(6)
pp.plot(h_av)

#pp.figure(7)
#pp.plot(spred)


pp.figure(8)
pp.plot(xpred[:,:,0],'r')
pp.plot(xact,'b')

ploss_hist=np.asarray(ploss_hist)
pp.figure(9)
pp.plot(ploss_hist)

#pp.figure(6)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,0],c='k',s=5)

#pp.figure(5)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,1],color=zip(np.zeros(len(w_hist)),np.zeros(len(w_hist)),np.ones(len(w_hist)),w_hist[:,i]))


pp.show()

