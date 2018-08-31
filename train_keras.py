import tensorflow as tf
from keras.datasets import boston_housing
import keras
import numpy as np
import pandas
import matplotlib.pyplot as plt



# prepare data
(train_data, train_label), (test_data, test_label) = boston_housing.load_data()  # (404, 13), (404, ), (102, 13), (102, )

# Shuffle 打乱数据顺序
order = np.argsort(np.random.random(train_label.shape))
train_data = train_data[order]
train_labels = train_label[order]


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df=pandas.DataFrame(train_data,columns=column_names)
df.head()

#normalize
mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std

#build model
def build_model():
    model=keras.Sequential()
    model.add(keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64,activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))

    optimizer=tf.train.RMSPropOptimizer(0.001)

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])
    return model
model=build_model()
model.summary()


#train model
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):#logs!=None,None类型is not callable
    if epoch % 100 == 0: print('')
    print('>', end='')

earlyStopping=keras.callbacks.EarlyStopping('val_loss',patience=20)

history=model.fit(x=train_data,y=train_labels,epochs=500,verbose=0,
                  callbacks=[earlyStopping,PrintDot()],validation_split=0.2)



def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs Error [1000$]')
    plt.plot(history.epoch,np.array(history.history['mean_absolute_error']),label='Train loss')
    plt.plot(history.epoch,np.array(history.history['val_mean_absolute_error']),label='val_loss')
    plt.legend()
    plt.ylim([0,8])


plot_history(history)
plt.show()

#eval
[loss, mae] = model.evaluate(test_data, test_label, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

#prediction
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_label, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()


error = test_predictions - test_label
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()