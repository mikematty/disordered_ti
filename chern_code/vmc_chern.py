import numpy as np

class SimState(object):
  def __init__(self,l,ne,wvfns):
    self.l, self.size, self.ne = l, 2*l*l, ne
    assert((wvfns.shape[0] == ne) and (wvfns.shape[1] == self.size))
    self.rlspc = np.random.permutation(self.size)
    self.wvfn = wvfns[:,self.rlspc]
    self.wvfn_inv = np.array(np.linalg.inv(self.wvfn[:,:self.ne]))

  def step(self):
    e_to_move = np.random.randint(0,self.ne)
    moveto = np.random.randint(self.ne,self.size)
    new_cols = [i if i != e_to_move else moveto for i in range(self.ne)]
    r = det_ratio(self.wvfn_inv,self.wvfn[:,new_cols],e_to_move)
    if min(1.0,np.power(np.abs(r),2.0)) > np.random.random():
      cols = [i for i in range(self.size)]
      cols[e_to_move],cols[moveto] = cols[moveto],cols[e_to_move]
      self.wvfn = self.wvfn[:,cols]
      self.rlspc = self.rlspc[cols]
      update_inverse(self.wvfn_inv,self.wvfn,r,e_to_move)
      return True
    return False
      

def det_ratio(wvfn_i_inv,wvfn_f,col):
  return sum(wvfn_i_inv[col,row]*wvfn_f[row,col] for row in range(wvfn_f.shape[0]))

def update_inverse(wvfn_i_inv, wvfn_f, r, col):
  size = wvfn_f.shape[0]
  for i in range(size):
    if i != col:
      beta_i = sum(wvfn_i_inv[i,k]*wvfn_f[k,col] for k in range(size))/r
      wvfn_i_inv[i,:] = wvfn_i_inv[i,:] - beta_i*wvfn_i_inv[col,:]
  wvfn_i_inv[col,:] = wvfn_i_inv[col,:]/r
  return

# does c^+_i c_j
def two_point(sim,i,j):
  if ((not i in sim.rlspc[:sim.ne])or(i == j) and (j in sim.rlspc[sim.ne:])):
    icol,jcol = list(sim.rlspc).index(i), list(sim.rlspc).index(j)
    cols = [x if x != jcol else icol for x in range(sim.ne)]
    # TODO: Wait can't I do this with the SMW trick?
    res = np.linalg.det(sim.wvfn[:,cols])/np.linalg.det(sim.wvfn[:,:sim.ne])
    #print(res)
    return res
  else: return 0+0j


