import numpy as np
from matplotlib import pyplot as pp

from SCLmodel import SCLmodel




nx=2
ns=2
nh=2
npcl=20

nsamps=10
lrate=1e-3

model=SCLmodel(nx, ns, nh, npcl, xvar=0.1)


x=T.fvector()

#norm, eng, ssmp, sprd, Wx, updates0=model.forward_filter_step(x)
#norm, eng, updates0=model.forward_filter_step(x)
#inference_step=theano.function([x],[norm,eng,ssmp,sprd,Wx],updates=updates0,allow_input_downcast=True)
#inference_step=theano.function([x],[norm,eng],updates=updates0,allow_input_downcast=True)
hsmps, updates0=model.forward_filter_step(x)
inference_step=theano.function([x],hsmps,updates=updates0,allow_input_downcast=True)

ess=model.get_ESS()
get_ESS=theano.function([],ess)

updates1=model.resample()
resample=theano.function([],updates=updates1)

x1=T.fvector(); x2=T.fvector()
lr=T.fscalar(); nsmps=T.lscalar()

nrg, updates2 = model.update_params(x1, x2, nsmps, lr)
learn_step=theano.function([x1,x2,nsmps,lr],[nrg],updates=updates2,allow_input_downcast=True)

nps=T.lscalar()
sps, xps, hs, updates3 = model.simulate_forward(nps)
predict=theano.function([nps],[sps,xps,hs],updates=updates3,allow_input_downcast=True)

theta=0.05
vec=np.ones(2)
M1=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]],dtype='float32')
M2=np.asarray([[np.cos(theta/3.0),-np.sin(theta/3.0)],[np.sin(theta/3.0),np.cos(theta/3.0)]],dtype='float32')
W=np.asarray(np.random.randn(2,2),dtype='float32')
c=np.asarray(np.random.randn(2),dtype='float32')

dt=0.05
nt=100000


x_hist=[]
h_hist=[]
v_hist=[]
e_hist=[]
s_hist=[]
r_hist=[]
l_hist=[]
w_hist=[]

resample_counter=0
learn_counter=0

for i in range(nt):
	if vec[0]+np.random.randn(1)/100.0>0:
		M=M1
	else:
		M=M2
	vec=np.dot(M,vec)+np.random.randn(2)*0.01
	x=np.dot(W,vec)+c+np.random.randn(2)*0.1
	v_hist.append(vec)
	x_hist.append(x)
	#normalizer,energies,ssamps,spreds,WTx=inference_step(vec)
	#normalizer,energies=inference_step(x)
	h_samps=inference_step(x)
	
	#pp.scatter(ssamps[:,0],ssamps[:,1],color='b')
	#pp.scatter(spreds[:,0],spreds[:,1],color='r')
	#pp.scatter(WTx[0],WTx[1],color='g')
	
	#pp.hist(energies,20)
	#pp.show()
	
	#print h_samps
	
	ESS=get_ESS()
	
	learn_counter+=1
	resample_counter+=1
	
	if resample_counter>0 and learn_counter>20:
		energy=learn_step(x_hist[-2],x_hist[-1],nsamps, lrate)
		e_hist.append(energy)
		learn_counter=0
		l_hist.append(1)
		lrate=lrate*0.9993094
	else:
		l_hist.append(0)
	
	if i%1000==0:	
		#print normalizer
		print ESS
		print i
		print model.M.get_value()
		print model.W.get_value()
		print model.A.get_value()
		print model.mu.get_value()
		print model.c.get_value()
		print model.b.get_value()
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
		print model.b.get_value()
		print model.M.get_value()
		print model.W.get_value()
		break


