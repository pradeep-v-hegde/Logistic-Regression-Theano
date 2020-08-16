import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import theano
import theano.tensor as T
from sklearn.metrics import accuracy_score

X_df = pd.read_csv('/Users/pradeep/Jupyter Notebooks/datasets_IRIS.csv')
Y = X_df[['species']]
X = X_df.loc[:,X_df.columns[0:4]]

X = StandardScaler().fit_transform(X)

Y = OneHotEncoder(sparse=False).fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=23)

x = T.dmatrix('x')
w = T.dmatrix('w')
b = T.dvector('b')
y = T.dmatrix('y')

z = theano.dot(x, w.T) + b
#y_hat = T.nnet.softmax(z)
y_hat = T.exp(z)/T.exp(z).sum()

fwd_calc_z = theano.function(inputs=[x, w, b], outputs=z)
fwd_calc_softmax = theano.function(inputs=[z], outputs=y_hat)
xe_loss = T.mean((-y*T.log(y_hat+1e-9)))

calc_xe_loss = theano.function(inputs=[y_hat, y], outputs=xe_loss)

dz = T.grad(xe_loss, z)

dw = T.grad(xe_loss, w)

db = T.grad(xe_loss, b)

calc_dz = theano.function(inputs=[x,w,b,y], outputs=dz)

calc_dw = theano.function(inputs=[x,w,b,y], outputs=dw)

calc_db = theano.function(inputs=[x,w,b,y], outputs=db)


def update_weights(y, y_hat, w, x, b, learning_rate):
    dz = calc_dz(x,w,b,y)
    dw = calc_dw(x,w,b,y)
    db = calc_db(x,w,b,y)
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return [w, b, dz, dw, db]


def predict(x, w, b):
    y_hat = fwd_calc_softmax(fwd_calc_z(x,w,b))
    return y_hat


epoch = 200
print_freq = 100
batch_size = 1

#W = np.array([[0.94433152, 0.09602482, 0.77574389, 0.61971023],
#       [0.72381748, 0.73687407, 0.56327551, 0.22356149],
#       [0.98768556, 0.49652208, 0.78874629, 0.22600592]])

#B = np.array([0.,0.,0.])

W = np.random.uniform(0,1,(Y.shape[1],X.shape[1]))
B = np.random.uniform(0,1,(Y.shape[1]))



for e in range(epoch):
    for i in range(X_train.shape[0]):
    #for i in range(2):
        #print("\n\n")
        #print("X ", X_train[i:i+1,:])
        #print("W ", W)
        #print("\n")
        Z = fwd_calc_z(X_train[i:i+batch_size,:], W, B)
        #print("Z ", Z)
        Y_hat = fwd_calc_softmax(Z)
        #print("Y_hat", Y_hat)
        #print("Y ", Y_train[i:i+1,:])
        #print("XE Loss ", calc_xe_loss(Y_hat, Y_train[i:i+1,:]))
        W, B, dZ, dW, dB = update_weights(Y_train[i:i+batch_size,:], Y_hat, W, X_train[i:i+batch_size,:], B, learning_rate=0.01)
        #print("Gradient of Z", dZ)
        #print("Gradient of W", dW)
        if(i%print_freq == 0):
            #calc train and test loss
            train_loss = calc_xe_loss(Y_train, predict(X_train, W, B))
            train_acc = accuracy_score(np.argmax(Y_train,axis=1), np.argmax(predict(X_train, W, B), axis=1))
            test_loss = calc_xe_loss(Y_test, predict(X_test, W, B))
            test_acc = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(predict(X_test, W, B), axis=1))
            print("After %d epoch %d iterations, train_loss %f,test_loss %f, train_acc %f, test_acc %f"%(e,i,train_loss, test_loss, train_acc, test_acc))
            
    
