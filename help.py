import numpy as np
data=np.array([(1,),
      (2,),
      (3,),
      (4,),
      (5,),
      (6,),
      (7,),])
list=np.array([12.,54.,3.,1.,50.,11.,21.,])

order = np.argsort(np.random.random(list.shape))
train_data = data[order]
train_labels = list[order]
print(train_data)

print(train_labels)