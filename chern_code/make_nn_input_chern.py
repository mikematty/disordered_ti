import numpy as np, progressbar
from ed_chern import *
from vmc_chern import *
from qlt import *

def main():
  # do everythin ~in situ~ for now lol
  # Hamiltonian paramters
  n, t_nn, t_nnn, m, w = 8, 4.0, 1.0, 1.0, 1

  # VMC parameters (ne is half filling)
  ne, n_steps, n_equil, n_collect = n*n, int(2e3), int(1e4), int(2e2) #4e4
  n_configs = 1000 # number of disorder configurations
  
  # I/O parameters
  outf = "disorder_topo.txt" #"test_disorder_transition/w_6.txt"

  ###### Ideally shouldn't have to touch anything below here ##################

  # precompute triangle indices for QLT to aid VMC sampling
  triangles_inds = qlt_triangles(n)
  triangles_ints = [ [x for x in map(lambda trig: ind_to_int(trig,n), \
                   triangles_inds[i])] for i in range(len(triangles_inds))]
  bar = progressbar.ProgressBar(maxval = 50, \
        widgets=[progressbar.Bar('=','[',']'),' ',progressbar.Percentage()])
  bar.start()
  nn_input=np.zeros(((n_steps*n_configs)//n_collect,\
                     len(triangles_ints)),dtype='complex')
  for config in range(n_configs):
    # get the eigenvectors of the Hamiltonian specified by the above params
    ham = haldane_ham(n,m,t_nn,t_nnn,w)
    nrgs,sp_wvfns = diagonalize(ham)
    sp_wvfns = np.transpose(sp_wvfns)

    # variationally sample the wave function and collect qlt data
    sims = [SimState(n,ne,sp_wvfns[:n*n,:]) for sim_ind in range(3)]
    for i in range(n_equil): 
      for sim in sims: sim.step()
    bar = progressbar.ProgressBar(maxval = 50, \
          widgets=[progressbar.Bar('=','[',']'),' ',progressbar.Percentage()])
    bar.start()
    for nrun in range(n_steps):
      for sim in sims: sim.step()
      #### print progress bar ######
      progress = (nrun+(config*n_steps)) / (n_steps*n_configs)
      pos = int(round(50*progress))
      bar.update(pos)
      ##############################
      if(nrun % n_collect == 0):
        qlt_data = []
        for trig in triangles_ints:
          # TODO: wait can I precompute all possible two point functions?
          qlt_data += [np.prod([two_point(sim,i,j) for (sim,i,j) in \
                       [(sims[0],trig[0],trig[1]),(sims[1],trig[1],trig[2]),\
                        (sims[2],trig[2],trig[0])]])]
        nn_input[((nrun+config*n_steps))//n_collect,:] = qlt_data

  # You did it!
  np.savetxt(outf,nn_input)
  print()
  return

if __name__ == '__main__': main()    
