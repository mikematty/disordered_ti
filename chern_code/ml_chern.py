import sys, os, pickle, numpy as np, matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Model
from keras.layers import Input, Dense
sys.stderr = stderr

def supervised(train_data,train_labels,test_data,test_labels,n_epochs):
  input_img = Input(shape=(train_data.shape[1],))
  hidden = Dense(10,activation='relu')(input_img)
  output = Dense(2,activation='softmax')(hidden)

  neural_net = Model(input_img, output)
  neural_net.compile(optimizer='adadelta',loss='binary_crossentropy',
                     metrics = ['accuracy'])
  history=neural_net.fit(train_data,train_labels,epochs=n_epochs,batch_size=300,shuffle=True,
                 validation_data=(test_data,test_labels),verbose=0)
  scores = neural_net.evaluate(test_data,test_labels,verbose=0)
  print("\n%s: %.2f%%" % (neural_net.metrics_names[1], scores[1]*100))
  return neural_net, history

def load_qlt_data(fname,chunk_size = 1):
  data = np.loadtxt(fname,dtype='complex')
  if chunk_size > 1:
    print("Not implemented yet...")
    assert(False)

  real_data = np.zeros((data.shape[0],2*data.shape[1]),dtype='float64')
  for i in range(data.shape[0]):
    real_data[i,:] = [data[i,j//2].imag if (j%2) else data[i,j//2].real 
                      for j in range(real_data.shape[1])]

  return real_data

def load_classification_data(classify_dir, transform = lambda x: x):
  def parser(f):
    m = float(f[f.rfind("_")+1:f.find(".txt")])
    return m

  data, params = [], []
  dir_data = os.listdir(classify_dir)
  for f in filter(lambda x: ".txt" in x, dir_data):
    f_data = transform(np.loadtxt(classify_dir+f,dtype='complex'))
    data += [f_data]
    params += [parser(f)]*f_data.shape[0]
  data = np.vstack(data)
  real_data = np.zeros((data.shape[0],2*data.shape[1]),dtype='float64')
  for i in range(data.shape[0]):
    real_data[i,:] = [data[i,j//2].imag if (j%2) else data[i,j//2].real 
                      for j in range(real_data.shape[1])]

  return real_data, np.hstack(params)

def param_average(data,params,indices = None):
  average_dict = {}
  for datum, param in zip(data,params):
    key = param if indices is None else\
          tuple([param[i] for i in sorted(indices)])
    average_dict[key] = average_dict.get(key,[])+[datum]
  averaged = [(k,sum(v)/float(len(v))) for k,v in average_dict.items()]
  return np.vstack([x[1] for x in averaged]),np.array([x[0] for x in averaged])

def main():
  # Load the training data, partition into train/test
  topo_data = load_qlt_data("disorder_topo.txt")
  trivial_data = load_qlt_data("disorder_trivial.txt")
  topo_train,topo_test = topo_data[:9000],topo_data[9000:]
  triv_train,triv_test = trivial_data[:9000],trivial_data[9000:]
  train_data = np.vstack([topo_train,triv_train])
  train_labels = np.vstack([[1,0] for i in range(len(topo_train))]+
                           [[0,1] for i in range(len(triv_train))])
  test_data = np.vstack([topo_test,triv_test])
  test_labels = np.vstack([[1,0] for i in range(len(topo_test))]+
                          [[0,1] for i in range(len(triv_test))])

  # Do supervised learning
  net,history = supervised(train_data,train_labels,test_data,test_labels,300)

  # Load classification data and feed it through network
  class_data = "test_disorder_transition/"
  classification_data,params = load_classification_data(class_data)
  classified = net.predict(classification_data)

  # Plot results
  classified, params = param_average(classified,params)
  plt.plot(history.history['loss'])
  plt.plot(history.history['acc'])
  plt.figure(2)
  plt.plot(params,classified[:,0],marker='o',linestyle='')
  plt.show()

if __name__ == '__main__': main()
