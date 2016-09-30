##### Import simfunctions

from simfunctions import *



#############  Simulation on Real Data ################

# Load data set
# This file './real_data.json' is a list of dictionaries.  Each dictionary contains:
## The target user id ('target_id')
## The target's new account id ('target_new_acct')
## The dataset(s) this account pair belonged to in the logistic regression model ('dataset')
## The cluster ID from the original clustering done in R ('cluster_id')
## The ordered refollow probabilities ('phi0') for each former friend
## The ordered number of followers for each former friend ('n0')
## A ground truth binary vector ('ground_truth') that takes value 1 at position i if the target refollowed former friend i with the new account, otherwise 0.

f=open('./real_data.json')
rd=json.load(f)
f.close()

results=[]


nm=5000
i=0
for tp in rd: #target account pair in the loaded data
	i+=1
	reslt={}
	reslt['Run']='Real.Data.'+str(i)
	reslt['target_id']=tp['target_id']
	reslt['target_new_acct']=tp['target_new_acct']
	## Generate policies
	### Deterministic Policies
	d_policies={
	'op': generate_optimal_pi(tp['n0'],tp['phi0'],N_M=nm), #Optimal policy
	'gp': generate_greedy_pi(tp['n0'],tp['phi0'],N_M=nm), #Greedy policy
	'mxphi_block': generate_max_phi_block(tp['n0'],tp['phi0'],N_M=nm), #Choose the former friend with highest probability (phi) at each step and query to exhaustion
	'mxphi_stage': generate_max_phi_stage(tp['n0'],tp['phi0'],N_M=nm), #Choose the former friend with highest probability (phi) at each step.
	'mnn': generate_min_n_pi(tp['n0'],N_M=nm) #Choose a former friend with the minimum number of unqueried followers
	}
	### Random Policies
	#rb=generate_random_block(tp['n0']) #Choose a former friend randomly and query to exhaustion
	#rs=generate_random_stage(tp['n0']) #Choose a former friend randomly at each stage.
	##
	## Generate rho_0
	#rho_0=rho0=get_rho_0=rho0(tp['phi0'])
	rho0=1
	## Analysis and Simulation using deterministic policies
	### Theoretical Analysis of Real Data
	#### Expected policy costs
	reslt['epc_op']=expected_policy_cost(tp['n0'],tp['phi0'],d_policies['op'],rho_0=rho0,N_M=nm)
	reslt['epc_gp']=expected_policy_cost(tp['n0'],tp['phi0'],d_policies['gp'],rho_0=rho0,N_M=nm)
	reslt['epc_mxpb']=expected_policy_cost(tp['n0'],tp['phi0'],d_policies['mxphi_block'],rho_0=rho0,N_M=nm)
	reslt['epc_mxps']=expected_policy_cost(tp['n0'],tp['phi0'],d_policies['mxphi_stage'],rho_0=rho0,N_M=nm)
	reslt['epc_mnn']=expected_policy_cost(tp['n0'],tp['phi0'],d_policies['mnn'],rho_0=rho0,N_M=nm)
	#### Expected policy costs given ground truth data
	reslt['epc_fi_op']=expected_policy_cost(tp['n0'],tp['ground_truth'],d_policies['op'],rho_0=1,N_M=nm)
	reslt['epc_fi_gp']=expected_policy_cost(tp['n0'],tp['ground_truth'],d_policies['gp'],rho_0=1,N_M=nm)
	reslt['epc_fi_mxpb']=expected_policy_cost(tp['n0'],tp['ground_truth'],d_policies['mxphi_block'],rho_0=1,N_M=nm)
	reslt['epc_fi_mxps']=expected_policy_cost(tp['n0'],tp['ground_truth'],d_policies['mxphi_stage'],rho_0=1,N_M=nm)
	reslt['epc_fi_mnn']=expected_policy_cost(tp['n0'],tp['ground_truth'],d_policies['mnn'],rho_0=1,N_M=nm)
	### Real Data Simulations
	print("Starting deterministic policy simulations")
	for plcy in d_policies.keys():
		#Ground Truth Simulations
		n_stages=[]
		for j in range(100): 
			n_stages.append(simulate(tp['n0'],tp['ground_truth'],d_policies[plcy],rho_0=1,N_M=nm))
		reslt[plcy+'_sim']=n_stages
		#Policy simulations based on reconnection probabilities
		n_stages=[]
		for k in range(100): #100
			n_stages.append(simulate(tp['n0'],tp['phi0'],d_policies[plcy],rho_0=rho0,N_M=nm))
		reslt[plcy+'_noinfo_sim']=n_stages
	## Simulation using random policies
	rb_epc_fi=[]
	rs_epc_fi=[]
	rb_epc_ni=[]
	rs_epc_ni=[]
	rb_stages_fi=[]
	rs_stages_fi=[]
	rb_stages_ni=[]
	rs_stages_ni=[]
	#Random policy simulations
	print("Starting random policy simulations")
	for j in range(500): #500
		#Policy generation
		rb=generate_random_block(tp['n0'],N_M=nm)
		rs=generate_random_stage(tp['n0'],N_M=nm)
		#Ground truth expected cost calculations
		rb_epc_fi.append(expected_policy_cost(tp['n0'],tp['ground_truth'],rb,rho_0=1,N_M=nm))
		rs_epc_fi.append(expected_policy_cost(tp['n0'],tp['ground_truth'],rs,rho_0=1,N_M=nm))
		#Expected costs based on reconnection probabilities
		rb_epc_ni.append(expected_policy_cost(tp['n0'],tp['phi0'],rb,rho_0=rho0,N_M=nm))
		rs_epc_ni.append(expected_policy_cost(tp['n0'],tp['phi0'],rs,rho_0=rho0,N_M=nm))
		#Simulations based on ground truth
		rb_stages_fi.append(simulate(tp['n0'],tp['ground_truth'],rb,rho_0=1,N_M=nm))
		rs_stages_fi.append(simulate(tp['n0'],tp['ground_truth'],rs,rho_0=1,N_M=nm))
		#Simulations based on reconnection probabilities
		rb_stages_ni.append(simulate(tp['n0'],tp['phi0'],rb,rho_0=rho0,N_M=nm))
		rs_stages_ni.append(simulate(tp['n0'],tp['phi0'],rs,rho_0=rho0,N_M=nm))
	reslt['rb_epc_fi']=rs_epc_fi
	reslt['rs_epc_fi']=rb_epc_fi
	reslt['rb_epc_ni']=rs_epc_ni
	reslt['rs_epc_ni']=rb_epc_ni
	reslt['rb_stages_fi']=rb_stages_fi
	reslt['rs_stages_fi']=rs_stages_fi
	reslt['rb_stages_ni']=rb_stages_ni
	reslt['rs_stages_ni']=rs_stages_ni
	results.append(reslt)
	print(str(i)+'/'+str(len(rd))+' complete.')

f=open('fake_results.json','w')
try:
	json.dump(results,f)
except:
	pass
f.close()



############# Notional Data Sets ############

