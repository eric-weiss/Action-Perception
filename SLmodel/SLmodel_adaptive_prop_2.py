import numpy as np
from matplotlib import pyplot as pp

import scipy.linalg as spla

import math

import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class SLmodel():
	
	#This version adapts the proposal distribution by keeping a running
	#estimate of the exact posterior covariance, parametrized as the 
	#matrix CC'
	
	def __init__(self, nx, ns, nh, npcl, xvar=1.0):
		
		#for this model I assume one linear generative model and a 
		#combination of nh linear dynamical models
		
		#generative matrix
		init_W=np.asarray(np.random.randn(nx,ns)/10.0,dtype='float32')
		#init_W=np.asarray(np.eye(2),dtype='float32')
		
		#always normalize the columns of W to be unit length
		init_W=init_W/np.sqrt(np.sum(init_W**2,axis=0))
		
		#observed variable means
		init_c=np.asarray(np.zeros(nx),dtype='float32')
		
		#dynamical matrices
		#init_M=np.asarray(np.random.randn(ns,ns*nh)/2.0,dtype='float32')
		init_M=np.asarray((np.tile(np.eye(ns),(1,nh))),dtype='float32')
		
		#state-variable variances
		#(covariance matrix of state variable noise assumed to be diagonal)
		init_b=np.asarray(np.ones(ns)*10.0,dtype='float32')
		
		#Switching parameter matrix
		init_A=np.asarray(np.zeros((ns,nh)),dtype='float32')
		
		#priors for switching variable
		init_ph=np.asarray(np.zeros(nh),dtype='float32')
		
		self.W=theano.shared(init_W)
		self.c=theano.shared(init_c)
		self.M=theano.shared(init_M)
		self.b=theano.shared(init_b)
		self.A=theano.shared(init_A)
		self.ph=theano.shared(init_ph)
		
		#square root of covariance matrix of proposal distribution
		#initialized to the true root covariance
		init_cov_inv=np.dot(init_W.T, init_W)/(xvar**2) + np.eye(ns)*np.exp(-init_b)
		init_cov=spla.inv(init_cov_inv)
		init_C=spla.sqrtm(init_cov)
		init_C=np.asarray(np.real(init_C),dtype='float32')
		
		
		init_s_now=np.asarray(np.zeros((npcl,ns)),dtype='float32')
		init_h_now=np.asarray(np.zeros((npcl,nh)),dtype='float32')
		init_h_now[:,0]=1.0
		init_weights_now=np.asarray(np.ones(npcl)/float(npcl),dtype='float32')
		
		init_s_past=np.asarray(np.zeros((npcl,ns)),dtype='float32')
		init_h_past=np.asarray(np.zeros((npcl,nh)),dtype='float32')
		init_h_past[:,0]=1.0
		init_weights_past=np.asarray(np.ones(npcl)/float(npcl),dtype='float32')
		
		
		
		self.C=theano.shared(init_C)
		
		#this is to help vectorize operations
		self.sum_mat=T.as_tensor_variable(np.asarray((np.tile(np.eye(ns),nh)).T,dtype='float32'))
		
		self.s_now=theano.shared(init_s_now)
		self.h_now=theano.shared(init_h_now)
		self.weights_now=theano.shared(init_weights_now)
		
		self.s_past=theano.shared(init_s_past)
		self.h_past=theano.shared(init_h_past)
		self.weights_past=theano.shared(init_weights_past)
		
		self.xvar=np.asarray(xvar,dtype='float32')
		
		self.nx=nx		#dimensionality of observed variables
		self.ns=ns		#dimensionality of latent variables
		self.nh=nh		#number of (linear) dynamical modes
		self.npcl=npcl	#numer of particles in particle filter
		
		
		#for ease of use and efficient computation (these are used a lot)
		self.CCT=T.dot(self.C, self.C.T)
		self.cov_inv=T.dot(self.W.T, self.W)/(self.xvar**2) + T.eye(ns)*T.exp(-self.b)
		
		
		self.theano_rng = RandomStreams()
		
		self.params=				[self.W, self.M, self.b, self.A, self.c, self.ph]
		self.rel_lrates=np.asarray([  0.1,    1.0,    1.0,   10.0,    1.0,     1.0]   ,dtype='float32')
		
		self.meta_params=     [self.C]
		self.meta_rel_lrates=[   1.0  ]
	
	
	def sample_proposal_s(self, s, h, xp):
		
		s_pred=self.get_prediction(s, h)
		
		n=self.theano_rng.normal(size=T.shape(s))
		
		mean_term=T.dot((xp-self.c), self.W)/(self.xvar**2) + s_pred*T.exp(-self.b)
		prop_mean=T.dot(mean_term, self.CCT)
		
		s_prop=prop_mean + T.dot(n, self.C)
		
		#I compute the term inside the exponent for the pdf of the proposal distrib
		prop_term=-T.sum(n**2)/2.0
		
		return T.cast(s_prop,'float32'), T.cast(s_pred,'float32'), T.cast(prop_term,'float32'), prop_mean
	
	
	def calc_h_probs(self, s):
		
		#this function takes an np by ns matrix of s samples
		#and returns an nh by np set of h probabilities
		
		exp_terms=T.dot(s, self.A) + T.reshape(self.ph,(1,self.nh))
		
		#re-centering for numerical stability
		exp_terms_recentered=exp_terms-T.max(exp_terms,axis=1)
		
		#exponentiation and normalization
		rel_probs=T.exp(exp_terms)
		probs=rel_probs.T/T.sum(rel_probs, axis=1)
		
		return probs.T
	
	
	def forward_filter_step(self, xp):
		
		#need to sample from the proposal distribution first
		s_samps, s_pred, prop_terms, prop_means = self.sample_proposal_s(self.s_now,self.h_now,xp)
		
		updates={}
		
		#now that we have samples from the proposal distribution, we need to reweight them
		
		h_probs = self.calc_h_probs(s_samps)
		
		h_samps=self.theano_rng.multinomial(pvals=h_probs)
		
		recons=T.dot(self.W, s_samps.T) + T.reshape(self.c,(self.nx,1))
		
		x_terms=-T.sum((recons-T.reshape(xp,(self.nx,1)))**2,axis=0)/(2.0*self.xvar**2)
		s_terms=-T.sum(((s_samps-s_pred)*self.b)**2,axis=1)/2.0
		
		energies=x_terms+s_terms-prop_terms
		
		#to avoid exponentiating large or very small numbers, I 
		#"re-center" the reweighting factors by adding a constant, 
		#as this has no impact on the resulting new weights
		
		energies_recentered=energies-T.max(energies)
		
		alpha=T.exp(energies_recentered) #these are the reweighting factors
		
		new_weights_unnorm=self.weights_now*alpha
		normalizer=T.sum(new_weights_unnorm)
		new_weights=new_weights_unnorm/normalizer  #need to normalize new weights
		
		updates[self.h_past]=T.cast(self.h_now,'float32')
		updates[self.s_past]=T.cast(self.s_now,'float32')
		
		updates[self.h_now]=T.cast(h_samps,'float32')
		updates[self.s_now]=T.cast(s_samps,'float32')
		
		updates[self.weights_past]=T.cast(self.weights_now,'float32')
		updates[self.weights_now]=T.cast(new_weights,'float32')
		
		#return normalizer, energies_recentered, s_samps, s_pred, T.dot(self.W.T,(xp-self.c)), updates
		#return normalizer, energies_recentered, updates
		#return h_samps, updates
		return updates
	
	
	def proposal_loss(self,C):
		
		#calculates how far off self.CCT is from the true posterior covariance
		CCT=T.dot(C, C.T)
		prod=T.dot(CCT, self.cov_inv)
		diff=prod-T.eye(self.ns)
		tot=T.sum(T.sum(diff**2))  #frobenius norm
		
		return tot
	
	
	def prop_update_step(self, C_now, lr):
		
		loss=self.proposal_loss(C_now)
		gr=T.grad(loss, C_now)
		return [C_now-lr*gr]
	
	
	def update_proposal_distrib(self, n_steps, lr):
		
		#does some gradient descent on self.C, so that self.CCT becomes
		#closer to the true posterior covariance
		C0=self.C
		Cs, updates = theano.scan(fn=self.prop_update_step,
									outputs_info=[C0],
									non_sequences=[lr],
									n_steps=n_steps)
		
		updates[self.C]=Cs[-1]
		
		loss=self.proposal_loss(Cs[-1])
		
		#updates={}
		#updates[self.C]=self.prop_update_step(self.C,lr)
		#loss=self.proposal_loss(self.C)
		
		return loss, updates
	
	
	def get_prediction(self, s, h):
		
		s_dot_M=T.dot(s, self.M)  #this is np by nh*ns
		s_pred=T.dot(s_dot_M*T.extra_ops.repeat(h,self.ns,axis=1),self.sum_mat) #should be np by ns
		
		return T.cast(s_pred,'float32')
	
	
	def sample_joint(self, sp):
		
		t2_samp=self.theano_rng.multinomial(pvals=T.reshape(self.weights_now,(1,self.npcl))).T
		s2_samp=T.cast(T.sum(self.s_now*T.addbroadcast(t2_samp,1),axis=0),'float32')
		h2_samp=T.cast(T.sum(self.h_now*T.addbroadcast(t2_samp,1),axis=0),'float32')
		
		diffs=self.b*(s2_samp-sp)
		sqr_term=T.sum(diffs**2,axis=1)
		alpha=T.exp(-sqr_term)
		probs_unnorm=self.weights_past*alpha
		probs=probs_unnorm/T.sum(probs_unnorm)
		
		t1_samp=self.theano_rng.multinomial(pvals=T.reshape(probs,(1,self.npcl))).T
		s1_samp=T.cast(T.sum(self.s_past*T.addbroadcast(t1_samp,1),axis=0),'float32')
		h1_samp=T.cast(T.sum(self.h_past*T.addbroadcast(t1_samp,1),axis=0),'float32')
		
		return [s1_samp, h1_samp, s2_samp, h2_samp]
	
	
	def calc_mean_h_energy(self, s, h):
		
		#you give this function a set of samples of s and h,
		#it gives you the average energy of those samples
		
		
		exp_terms=T.dot(s, self.A) + T.reshape(self.ph,(1,self.nh))  #np by nh
		
		energies=T.sum(h*exp_terms,axis=1) - T.log(T.sum(T.exp(exp_terms),axis=1)) #should be np by 1
		
		energy=T.mean(energies)
		
		return energy
	
	
	def update_params(self, x1, x2, n_samps, lrate):
		
		#this function samples from the joint posterior and performs
		# a step of gradient ascent on the log-likelihood
		
		sp=self.get_prediction(self.s_past, self.h_past)
									
		#sp should be np by ns
		
		
		[s1_samps, h1_samps, s2_samps, h2_samps], updates = theano.scan(fn=self.sample_joint,
									outputs_info=[None, None, None, None],
									non_sequences=[sp],
									n_steps=n_samps)
		
		
		
		x1_recons=T.dot(self.W, s1_samps.T) + T.reshape(self.c,(self.nx,1))
		x2_recons=T.dot(self.W, s2_samps.T) + T.reshape(self.c,(self.nx,1))
		
		s_pred = self.get_prediction(s1_samps, h1_samps)
		
		
		hterm1=self.calc_mean_h_energy(s1_samps, h1_samps)
		#hterm2=self.calc_mean_h_energy(s2_samps, h2_samps)
		
		sterm=-T.mean(T.sum((self.b*(s2_samps-s_pred))**2,axis=1))/2.0
		
		#xterm1=-T.mean(T.sum((x1_recons-T.reshape(x1,(self.nx,1)))**2,axis=0)/(2.0*self.xvar**2))
		xterm2=-T.mean(T.sum((x2_recons-T.reshape(x2,(self.nx,1)))**2,axis=0)/(2.0*self.xvar**2))
		
		#energy = hterm1 + xterm1 + hterm2 + xterm2 + sterm -T.sum(T.sum(self.A**2))
		energy = hterm1 + xterm2 + sterm 
		
		gparams=T.grad(energy, self.params, consider_constant=[s1_samps, s2_samps, h1_samps, h2_samps])
		
		# constructs the update dictionary
		for gparam, param, rel_lr in zip(gparams, self.params, self.rel_lrates):
			#gnat=T.dot(param, T.dot(param.T,param))
			updates[param] = T.cast(param + gparam*lrate*rel_lr,'float32')
		
		return energy, updates
		
	
	def get_ESS(self):
		
		return 1.0/T.sum(self.weights_now**2)
	
	
	def resample_step(self):
		
		idx=self.theano_rng.multinomial(pvals=T.reshape(self.weights_now,(1,self.npcl))).T
		s_samp=T.sum(self.s_now*T.addbroadcast(idx,1),axis=0)
		h_samp=T.sum(self.h_now*T.addbroadcast(idx,1),axis=0)
		
		return T.cast(s_samp,'float32'), T.cast(h_samp,'float32')
	
	
	def resample(self):
		
		[s_samps, h_samps], updates = theano.scan(fn=self.resample_step,
												outputs_info=[None, None],
												n_steps=self.npcl)
		
		updates[self.s_now]=T.cast(s_samps,'float32')
		updates[self.h_now]=T.cast(h_samps,'float32')
		updates[self.weights_now]=T.cast(T.ones_like(self.weights_now)/T.cast(self.npcl,'float32'),'float32') #dtype paranoia
		
		return updates
	
	
	def simulate_step(self, s):
		
		s=T.reshape(s,(1,self.ns))
		#get h probabilities
		h_probs = self.calc_h_probs(s)
		
		#h_samp=self.theano_rng.multinomial(pvals=T.reshape(h_probs,(self.nh,1)))
		h_samp=self.theano_rng.multinomial(pvals=h_probs)
		
		sp=self.get_prediction(s,h_samp)
		
		xp=T.dot(self.W, sp.T) + T.reshape(self.c,(self.nx,1))
		
		return T.cast(sp,'float32'), T.cast(xp,'float32'), h_samp
		
	
	def simulate_forward(self, n_steps):
		
		
		s0=T.sum(self.s_now*T.reshape(self.weights_now,(self.npcl,1)),axis=0)
		s0=T.reshape(s0,(1,self.ns))
		[sp, xp, hs], updates = theano.scan(fn=self.simulate_step,
										outputs_info=[s0, None, None],
										n_steps=n_steps)
		
		return sp, xp, hs, updates



