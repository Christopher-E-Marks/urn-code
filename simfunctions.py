from __future__ import division
import json
import numpy as np

########### Least Informative Prior for rho0

def get_rho0(varphi0):
    varphi0=np.array(varphi0)
    return(1./(pow(np.prod(1-varphi0),-(np.prod(1-varphi0))/(1-np.prod(1-varphi0)))+(1-np.prod(1-varphi0))))



############ Policy functions ###########

##### Inputs:
###### N -- A vector containing the numbers of followers
###### varphi -- A vector of reconnection probabilities
###### N_M  -- The maximum number of results per query.

##Optimal policy

def generate_optimal_pi(N,varphi,N_M=5000):
    N=np.array(N)
    varphi=np.array(varphi)
    NJ=np.ceil(N/N_M).astype(int)-1 # This is the total number of queries minus 1 for each former friend
    first_order=np.minimum(N/(N_M*varphi)-(NJ+1)/2,
        [1/varphi[i]*(NJ[i]+1)-N_M/(2*N[i])*(NJ[i]+1)*NJ[i]-1 if NJ[i]>0 else 1 for i in range(len(NJ))])
    onepulls=[i for i in range(len(NJ)) if NJ[i]==0]
    zeropulls=[i for i in range(len(NJ)) if NJ[i]==-1]
    blocks=(N/(N_M*varphi)-(NJ+1)/2) >= [1/varphi[i]*(NJ[i]+1)-N_M/(2*N[i])*(NJ[i]+1)*NJ[i]-1 if NJ[i]>0 else 1 for i in range(len(NJ))]
    second_order=(N*(1-varphi)/(varphi*(N-NJ*N_M)))
    pulls=np.zeros(len(N))
    values=first_order
    for i in onepulls:
        pulls[i]=1
        values[i]=second_order[i]
    for i in zeropulls:
        pulls[i]=2
        values[i]=np.inf
    policy=[]
    while any(pulls<2):
        choice=np.argmin(values)
        if pulls[choice]==0 and not blocks[choice]:
            values[choice]=second_order[choice]
            policy+=[choice]*NJ[choice] #NJ[choice could be 0]!
            pulls[choice]=1
        elif pulls[choice]==0 and blocks[choice]:
            values[choice]=np.inf
            policy+=[choice]*(NJ[choice]+1)
            pulls[choice]=2
        else:
            values[choice]=np.inf
            policy+=[choice]
            pulls[choice]=2
    return(policy)


#Greedy policy looking for most likely return

def generate_greedy_pi(N,varphi,N_M=5000):
    N=np.array(N)
    varphi=np.array(varphi)
    NJ=np.ceil(N/N_M).astype(int)-1
    omega_0=np.array([0 if i==0 else min(1,N_M/i) for i in N])
    first_order=varphi*omega_0
    #second_order=1-np.array([N[i]*(1-varphi[i])/(varphi[i]*(N[i]-NJ[i]*N_M)) if N[i]>0 else 1 for i in range(len(N))])
    second_order=(N-NJ*N_M)/(N-varphi*NJ*N_M)
    pulls=np.zeros(len(N))
    onepulls=[i for i in range(len(NJ)) if NJ[i]==0]
    zeropulls=[i for i in range(len(NJ)) if NJ[i]==-1]
    values=first_order
    for i in onepulls:
        pulls[i]=1
    policy=[]
    for i in zeropulls:
        pulls[i]=2
        values[i]=-np.inf
    while any(pulls<2):
        choice=np.argmax(values)
        if pulls[choice]==0:
            values[choice]=second_order[choice]
            policy+=[choice]*NJ[choice]
        else:
            values[choice]=-np.inf
            policy+=[choice]
        pulls[choice]+=1
    return(policy)


#Exhaust most probable former friend
#Pull each until complete

def generate_max_phi_block(N,varphi,N_M=5000):
    N=np.array(N)
    varphi=np.array(varphi)
    NJ=np.ceil(N/N_M).astype(int)
    first_order=varphi
    #phi_NJ=(varphi*(N-NJ*N_M))/(N-(NJ*N_M*varphi))
    #second_order=phi_NJ
    pulls=np.zeros(len(N))
    values=first_order
    policy=[]
    while any(pulls<1):
        choice=np.argmax(values)
        values[choice]=-np.inf
        policy+=[choice]*NJ[choice]
        pulls[choice]+=1
    return(policy)

#Most probable former friend greedy
def generate_max_phi_stage(N,varphi,N_M=5000):
    N=np.array(N)
    varphi=np.array(varphi)
    NJ=np.ceil(N/N_M).astype(int)
    first_order=varphi
    #phi_NJ=(varphi*(N-NJ*N_M))/(N-(NJ*N_M*varphi))
    #second_order=phi_NJ
    pulls=np.zeros(len(N))
    values=first_order
    zeropulls=[i for i in range(len(NJ)) if NJ[i]==0]
    for i in zeropulls:
        values[i]=-np.inf
    policy=[]
    while any(pulls<NJ):
        choice=np.argmax(values)
        policy+=[choice]
        pulls[choice]+=1
        if pulls[choice]<NJ[choice]:
            frac=(N[choice]-pulls[choice]*N_M)/N[choice]
            values[choice]=varphi[choice]*frac/(1-varphi[choice]+varphi[choice]*frac)
        else:
            values[choice]=-np.inf
    return(policy)


# Exhaust smallest degree former friend

def generate_min_n_pi(N,N_M=5000):
    N=np.array(N,dtype='float')
    NJ=np.ceil(N/N_M).astype(int)
    first_order=N
    #phi_NJ=(varphi*(N-NJ*N_M))/(N-(NJ*N_M*varphi))
    #second_order=phi_NJ
    pulls=np.zeros(len(N))
    values=first_order
    policy=[]
    while any(pulls<1):
        choice=np.argmin(values)
        values[choice]=np.inf
        policy+=[choice]*NJ[choice]
        pulls[choice]+=1
    return(policy)

# Random block

def generate_random_block(N,N_M=5000):
    N=np.array(N)
    NJ=np.ceil(N/N_M).astype(int)
    order=np.random.permutation(len(N))
    policy=[]
    for i in order:
        policy+=[i]*NJ[i]
    return(policy)

# Random stage

def generate_random_stage(N,N_M=5000):
    N=np.array(N)
    NJ=np.ceil(N/N_M).astype(int)
    policy=[]
    for i in range(len(NJ)):
        policy+=[i]*NJ[i]
    return(np.random.permutation(policy).tolist())



################## Policy Evaluations  ############

##Inputs:
### N: The vector of numbers of followers
### varphi: The ACTUAL refollow probabilities to be used for policy evaluation.
### policy: a vector of integers representing a sequence of former friend queries.
### rho_0: The initial existence probability.
### rho_bar:  The conditional existence probability threshold used as a TERMINATION CRITERION.
####  (All theoretical analysis done so far assumes rho_bar=0, implying that the search 
####   only terminates if the target has been found, or all queries are exhausted.)
### N_M: the maximum number of results returned for each query.

### Note: To obtain expected policy costs on ground truth data, set varphi and rho_0 to binary values representing ground truth.


#Evaluate (expected) cost of policy analytically

def expected_policy_cost(N,varphi,policy,rho_0=None,N_M=5000):
    if rho_0 is None:
        rho_0=get_rho0(varphi)
    varphi=np.array(varphi)
    N=np.array(N)
    NJ=np.ceil(N/N_M).astype(int)
    x=np.zeros(len(N))
    product=1
    terms=[]
    for i in range(len(policy)):
        pull=policy[i]
        if x[pull]<NJ[pull]-1:
            q_u=(N[pull]-varphi[pull]*(x[pull]+1)*N_M)/(N[pull]-varphi[pull]*x[pull]*N_M)
        else:
            q_u=(1-varphi[pull])*N[pull]/(N[pull]-varphi[pull]*x[pull]*N_M)
        x[pull]+=1
        product=product*(q_u)
        terms.append(product)
    return(len(policy)*(1-rho_0)+rho_0*sum(terms))


## Expected policy cost with rho_bar

def expected_policy_cost_bar(N,varphi,policy,rho_0=None,rho_bar=-0.00000001,N_M=5000):
    if rho_0 is None:
        rho_0=get_rho0(varphi)
    varphi=np.array(varphi)
    N=np.array(N)
    NJ=np.ceil(N/N_M).astype(int)
    x=np.zeros(len(N))
    product=1
    terms=[]
    rho=rho_0
    N_term=len(policy)
    for i in range(len(policy)):
        pull=policy[i]
        if x[pull]<NJ[pull]-1:
            q_u=(N[pull]-varphi[pull]*(x[pull]+1)*N_M)/(N[pull]-varphi[pull]*x[pull]*N_M)
        else:
            q_u=(1-varphi[pull])*N[pull]/(N[pull]-varphi[pull]*x[pull]*N_M)
        x[pull]+=1
        if rho <= rho_bar:
            if N_term > i:
                N_term=i
            terms.append((i)*product)
            product=0
        else:
            terms.append(product*q_u)
            product=product*(q_u)
        rho = rho_0 * product/(1-rho_0+rho_0*product)
    return(rho_0*sum(terms)+(1-rho_0)*(N_term))


################### Simulations

### Inputs are the same as before.
### Returns a single, simulated policy cost (not an expected cost) based on inputs.
### To simulate on ground truth data, set input varphi equal to ground truth binary refollow vector.


###### Simulate policy to termination.

def simulate(N,varphi,policy,rho_0=None,rho_bar=-0.000000001,N_M=5000):
    if rho_0 is None:
        rho_0=get_rho0(varphi)
    varphi=np.array(varphi)
    N=np.array(N)
    NJ=np.ceil(N/N_M).astype(int)
    x=np.zeros(len(N))
    product=1
    terms=[]
    p_exists=np.random.sample()
    if p_exists>rho_0:
        exists=False
    else:
        exists=True
    rho=rho_0
    i=0
    T=False
    while not T and i < len(policy) and rho>rho_bar:
        pull=policy[i]
        if x[pull]<NJ[pull]-1:
            q_u=(N[pull]-varphi[pull]*(x[pull]+1)*N_M)/(N[pull]-varphi[pull]*x[pull]*N_M)
        else:
            q_u=(1-varphi[pull])*N[pull]/(N[pull]-varphi[pull]*x[pull]*N_M)
        x[pull]+=1
        if exists:
            p_rand=np.random.sample()
            if p_rand > q_u: #Success!!!
                T=True
            else:
                i+=1
        else:
            i+=1
        product=product*(q_u)
        if product > 0 or rho_0 < 1:
            rho = rho_0 * product/(1-rho_0+rho_0*product)
    return(i)



############ Generate synthetic data from phi,n,rho:

def gen_syn_data(N,varphi,rho_0=None):
    if rho_0 is None:
        rho_0=get_rho0(varphi)
    varphi=np.array(varphi)
    N=np.array(N)
    p_exists=np.random.sample()
    if p_exists > rho_0:
        return(np.array(np.zeros(len(N)),dtype=int).tolist())
    else:
        p_friend=np.random.sample(len(N))
        return(np.array(p_friend<varphi,dtype=int))

