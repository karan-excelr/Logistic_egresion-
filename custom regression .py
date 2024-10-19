import numpy as np
class CustomLogisticRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None
        self.loss=[]

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def compute_log_loss(self,y_true,y_predicted):
        n_samples =len(y_true)
        loss=-(1/n_samples)*np.sum(y_true*np.log(y_predicted)+(1-y_true)*np.log(1-y_predicted))
        return loss
    
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for epoch in range(self.epochs):
            z=np.dot(x,self.weights)+self.bias
            y_predicted=self.sigmoid(z)

            dw =(1/n_samples)*np.dot(x.T,(y_predicted-y))
            db =(1/n_samples)*np.sum(y_predicted -y)
            
            self.weights -=self.learning_rate*dw
            self.bias -= self.learning_rate*db

            loss = self.compute_log_loss(y,y_predicted)
            self.loss.append(loss)

            if epoch%100==0:
                print(f"Epoch{epoch},log loss:{loss}")

    def predict(self,x):
        z=np.dot(x,self.weights)*self.bias
        y_predicted= self.sigmoid(z)
        return[1 if i > 0.5 else 0 for i in y_predicted]
    

if __name__=="__main__":
    x=np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]])
    y=np.array([0,0,0,1,1,1])
    model=CustomLogisticRegression(learning_rate=0.001,epochs=1000)
    model.fit(x,y)

    

        