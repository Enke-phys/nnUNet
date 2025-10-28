from tokenize import PlainToken
import torch
import torch.nn as nn
import torch.nn.functional as F

#Create structure of model
class Model(nn.Module):
    #Input layer (4 features of flowers) -> Hidden layer1 -> Hidden layer2 -> Output
    def __init__(self, input_features=4, hidden_features1=8, hidden_features2=4, output_features=3):
        super().__init__() #Erstellt ein Objekt
        #Define three layers
        self.fc1 = nn.Linear(input_features, hidden_features1)
        self.fc2 = nn.Linear(hidden_features1, hidden_features2)
        self.out = nn.Linear(hidden_features2, output_features)

#create function that moves everything forward
    def forward(self, x):
        #Start with layer 1
        x = F.relu(self.fc1(x))
        #Move to layer 2
        x = F.relu(self.fc2(x))
        #Move to output layer
        x = self.out(x)
        return x
        
#Pick a manual seed for randomization
torch.manual_seed(41)

#Create model instance (turn all on)
model = Model()

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
my_df = pd.read_csv(url)

#Change last colum from strings to integers
my_df["variety"] = my_df["variety"].replace({
    "Setosa": 0,
    "Versicolor": 1,
    "Virginica": 2
})

print(my_df.head())

#Input X and output y
X = my_df.drop("variety", axis=1)
y = my_df["variety"]

#Convert these to numpy arrays
X = X.to_numpy()
y = y.to_numpy()

#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set criterion of model to measure the error
criterion = nn.CrossEntropyLoss()
#Choose Adam optimizer, lr = learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train our model
epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
#print every 10 epochs
    if i % 10 == 0:
        print(f"Epoch: {i}, Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Graph it out
plt.plot(range(epochs),losses)
plt.ylabel("loss/error")
plt.xlabel("epoch")
#plt.show()

#Evaluate model on test data set
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test) #Find the loss or error
    print(f"Test Loss: {loss.item()}")

