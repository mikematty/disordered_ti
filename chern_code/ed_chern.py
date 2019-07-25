import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import eigh

# some nice utility lambdas
int_to_ind = lambda i,n: ((i//n)%n,i%n,i//(n*n))
gen_inds = lambda n: [((i//n)%n,i%n,i//(n*n)) for i in range(2*n*n)]
ind_to_int = lambda ind,n: ind[2]*n*n+ind[0]*n+ind[1]

# build exact Hamiltonian in real space basis
# non-interacting so just need to build single particle Ham
def build_honeycomb_ham(n,on_site,first_nn,second_nn):
  def is_nn(ind1,ind2):
    diff = np.subtract(ind2,ind1)
    diff = np.mod(diff,n)*np.sign(diff)
    return (diff[2] != 0) and \
           ( ((diff[0] == 0) and (abs(diff[1]) <= 1)) or 
             ((diff[1] == 0) and (abs(diff[0]) <= 1)) )
    
  def is_nnn(ind1,ind2):
    diff = np.subtract(ind2,ind1)
    diff = np.mod(diff,n)*np.sign(diff)
    return (diff[2] == 0) and \
           ( ((diff[0] == 0) and (abs(diff[1]) == 1)) or \
             ((diff[1] == 0) and (abs(diff[0]) == 1)) or \
             ((diff[0] == -1) and (diff[1] == 1)) or\
             ((diff[0] == 1) and (diff[1] == -1)) )

  inds = [(i,j,k) for k in [0,1] for i in range(n) for j in range(n)]
  ham = np.fromiter((\
    on_site(i)*(i==j) + first_nn(i,j)*is_nn(i,j) + second_nn(i,j)*is_nnn(i,j)\
    for i in inds for j in inds),dtype='complex128').\
    reshape((len(inds),len(inds)))
  return ham

# spin this off into a separate function in case I decide to make it more
# complicated later or whatever
def diagonalize(ham):
  return eigh(ham)

def haldane_ham(n,m,t_nn,t_nnn,w = 0):
  on_site = lambda ind: (m+np.random.normal(scale=w))*np.power(-1.0,ind[2])
  first_nn = lambda ind1,ind2: -t_nn
  def second_nn(ind1,ind2):
    diff = np.subtract(ind2,ind1)
    diff = np.mod(diff,n)*np.sign(diff)
    sign = (diff[0]==-diff[1])*diff[0]+(diff[0]*diff[1]==0)*(diff[1]-diff[0])
    sign *= np.power(-1,ind1[2])
    return -t_nnn*1j*sign

  return build_honeycomb_ham(n,on_site,first_nn,second_nn)

def main_ed_chern():
  n = 8
  m = 0
  t_nn = 1.0
  t_nnn = 0

  ham = haldane_ham(n,m,t_nn,t_nnn)
  nrgs,_ = diagonalize(ham)
  
  plt.hist(nrgs,bins=20)
  plt.show()
  for i in range(len(nrgs)): print(nrgs[i])

  return

if __name__ == '__main__': main_ed_chern()
