import numpy as np

def get_nns(ind,n):
  if ind[2] == 0:
    return [tuple(np.mod(np.add(ind,i),n))for i in [(0,-1,1),(0,0,1),(-1,0,1)]]
  elif ind[2] == 1:
    return [tuple(np.mod(np.add(ind,i),n))for i in[(0,1,-1),(0,0,-1),(1,0,-1)]]
  else: #something went terribly wrong...
    assert(False)
    return

def get_2nns(ind,n):
  return [tuple(np.mod(np.add(ind,i),n)) for i in \
          [(-1,0,0),(0,1,0),(1,-1,0),(1,0,0),(0,-1,0),(-1,1,0)]]


def qlt_triangles(n):
  inds = [(i,j,k) for k in [0,1] for i in range(n) for j in range(n)]
  triangles = []
  for ind in inds:
    nns = get_nns(ind,n)
    # three all - nn triangles
    triangles += [np.array([ind,nns[i],nns[j]]) for i in range(2) \
                                                for j in range(i+1,3)]
    # inter-cell triangle specified by ind
    triangles += [np.array(nns)]
    # all 2'nd nn triangle
    if ind[2] == 0:
      triangles += [np.array([ind,np.mod(np.add(ind,(0,1,0)),n),\
                              np.mod(np.add(ind,(-1,1,0)),n)])]
    else:
      triangles += [np.array([ind,np.mod(np.add(ind,(0,-1,0)),n),\
                              np.mod(np.add(ind,(1,-1,0)),n)])]
    # and finally, the six 1st-3rd triangles
    nnns_of_nns = [get_2nns(nn,n) for nn in nns]
    for i,nn in enumerate(nns):
      for nnn in nnns_of_nns[i]:
        for j in range(len(nnns_of_nns)):
          if (j != i) and (nnn in nnns_of_nns[j]) and (not (nnn in nns)):
            triangles += [np.array([ind,nns[i],nnn])]
    
  return triangles
    
        
  
