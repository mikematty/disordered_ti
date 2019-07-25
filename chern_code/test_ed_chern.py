import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_nrgs(ks,phi,t_nn,t_nnn,m):
  # nn lattice vectors for honeycomb lattice ordered so z.(a1 x a2) > 0
  a_s = [np.array([0,1/np.sqrt(3)]),np.array([-0.5,0.5/np.sqrt(3)]),
         np.array([0.5,0.5/np.sqrt(3)])]
  # nnn lattice vectors for honeycomb lattice ordered conveniently
  b_s = [a_s[1]-a_s[2],a_s[0]-a_s[2],a_s[0]-a_s[1],a_s[2]-a_s[1],\
         a_s[2]-a_s[0],a_s[1]-a_s[0]]
  
  f1 = lambda k: sum(np.cos(np.dot(k,b_s[i])) for i in range(len(b_s)))
  f2 = lambda k: sum(np.cos(np.dot(k,a_s[i])) for i in range(len(a_s)))
  f3 = lambda k: sum(np.sin(np.dot(k,a_s[i])) for i in range(len(a_s)))
  f4 = lambda k: sum(np.sin(np.dot(k,b_s[i])) for i in range(len(b_s)))
  nrgs = np.fromiter( (\
  0.5*(np.cos(phi)*t_nnn*f1(k)+nu*np.sqrt(4*(m*m+t_nn*t_nn*(f2(k)**2+f3(k)**2))+\
  t_nnn*(3*np.cos(phi)*f1(k)-4*np.sin(phi)*f4(k))*(4*m+t_nnn*(3*np.cos(phi)*\
  f1(k)-4*np.sin(phi)*f4(k))))) for nu in [1,-1] for k in ks), dtype = 'float')

  return nrgs

# n should be even pls
def gen_ks(n):
  #ks = [\
  #(2*2*np.pi/(3*n)) * ( i*np.array([1,np.sqrt(3)])+\
  #j*np.array([1,-np.sqrt(3)])+k*np.array([0,-2/np.sqrt(3)]) )\
  #for k in [0,1] for i in range(n//2) for j in range(n//2) ]
  #ks = [\
  #(2*2*np.pi/(3*n)) * ( i*np.array([np.sqrt(3),-1])+\
  #j*np.array([0,2])+k*np.array([-2/np.sqrt(3),0]) )\
  #for k in [0] for i in range(n//2) for j in range(n//2) ]
  ks = [np.array([i*4*np.pi/n,j*4*np.pi/n]) for i in range(n) for j in range(n)]
  return ks
  
def main():
  n = 40
  t_nn = 1
  t_nnn = 0
  m = 0
  phi = np.pi/2

  ks = gen_ks(n)
  nrgs = get_nrgs(ks,phi,t_nn,t_nnn,m)
  print(len(nrgs))
  #for i in range(len(nrgs)): print(sorted(nrgs)[i])

  plt.hist(nrgs,bins=30,density=True)
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(np.vstack(ks)[:,0],np.vstack(ks)[:,1],nrgs[:len(nrgs)//2])
  ax.scatter(np.vstack(ks)[:,0],np.vstack(ks)[:,1],nrgs[len(nrgs)//2:])
  plt.show()

if __name__ == '__main__': main()
