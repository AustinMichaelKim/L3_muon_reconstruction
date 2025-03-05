#!/usr/bin/env python
# coding: utf-8

# ### Dataset Preprocessing
# 
# reading dataset.
# 
# This dataset contains (1) 92113 samples of muon w/ 87 features.  (2) label array of 92113 samples  ( 1: true muon, 0: fake muon)
# 
# at first, we sample 100 muons for convinience.

# In[1]:


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


# Read data, no preprocessing
import pandas as pd
import pickle
import numpy as np




import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the .pkl file
with open('./QNN_using_qiskit/DYToLL_PU200_Spring23_NThltIter2FromL1_Barrel.pkl', 'rb') as file:
    data = pickle.load(file)


# Extract the DataFrame and the array from the tuple
df = data[0]  # The first element is the DataFrame : 92113 sample rows × 87 feature columns
array = data[1]  # The second element is the NumPy array : true labels for the 92113 samples

# Display the DataFrame
print(df)

# If you need to work with the array, you can do so separately
print(array)

# Slicing the data
# Randomly select 300 indices from the DataFrame for convinience
np.random.seed(0)
random_indices = np.random.choice(df.index, size=300, replace=False)

# Select the corresponding rows from the DataFrame
df_sampled = df.loc[random_indices]

# Select the corresponding labels from the array
array_sampled = array[random_indices]

# Display the sampled DataFrame and array
print(df_sampled)
print(array_sampled)

# Save the sampled data to an Excel file
# df_sampled.to_excel("./sampled_data.xlsx")

# Convert the sampled data to tensors
input_X = torch.tensor(df_sampled.values, dtype=torch.float32)
label_y = torch.tensor(array_sampled, dtype=torch.float32)

print(input_X)
print(label_y)

print(input_X.size())
print(label_y.size())


# In[3]:


# Read the data and preprocess it
import sys
import os

# Assuming your project structure as above, add the parent directory to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))



if project_root not in sys.path:
    sys.path.append(project_root)

# Now you should be able to import from BDT_model
from BDT_model.HLTIO import preprocess

import pandas as pd
import pickle

# Path to the original pickle file
pkl_path = "./QNN_using_qiskit/DYToLL_PU200_Spring23_NThltIter2FromL1_Barrel.pkl"


# Load the pickle file
with open(pkl_path, "rb") as file:
    data = pickle.load(file)

# Assume the first element of the tuple is the original DataFrame
df = data[0]

# Apply setClassLabel to compute the class labels
df = preprocess.setClassLabel(df)

# Compute the distance features (expd2hitl1tk1,..,expd2hitl1tk4)
# We use addAbsDist=False as in the readSeedTree flow to drop the d2hitl1tk variables.
df = preprocess.addDistHitL1Tk(df, addAbsDist=False)

# Define the list of required columns:
required_columns = [
    "expd2hitl1tk1",
    "expd2hitl1tk2",
    "expd2hitl1tk3",
    "dR_L1TkMuSeedP",
    "dPhi_L1TkMuSeedP",
    "tsos_qbp",
    "tsos_dydz",
    "tsos_dxdz",
    "tsos_err0",
    "tsos_err2",
    "tsos_err5",
    "y_label"  # class label from setClassLabel
]

# Check whether all required columns are present; if not, issue a warning.
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("Warning: The following required columns are missing:", missing)

# Create a new DataFrame with only the required columns.
df_final = df[required_columns].copy()

# Display the first few rows of the final DataFrame
print(df_final.head())

# Optionally, save the resulting DataFrame to an Excel file.
# df_final.to_excel("processed_data.xlsx", index=False)


# ### Data sampling and excluding nan value

# In[4]:


from IPython.display import display

# Exclude the samples with NaN values
df_final = df_final.dropna().copy()


# sampling 100 samples from the final DataFrame ( 100 samples w/ 11 features + 1 label)
df_final_sample = df_final.sample( n = 300 , replace=False, random_state=42).copy()


label_data = df_final_sample.iloc[:, -1] # extract y_label from the last column
input_data = df_final_sample.iloc[:, :-1] # extract X from the first 11 columns

X = torch.tensor(input_data.values, dtype=torch.float32)
y = torch.tensor(label_data.values, dtype=torch.float32)

split = int(0.8 * len(X))  # 80% train, 20% test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Move the data to the device ( GPU or CPU )
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


# ### Building Qunantum Circuit
# 
# Based on this dataset, build quantum circuit of 11 qubit

# In[ ]:


# import libraries

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from sklearn.metrics import accuracy_score

# qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp

# qiskit_aer
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi

# qiskit primitives
from qiskit_aer.primitives import EstimatorV2, SamplerV2

#qiskit qml related
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector





# In[6]:


from qiskit.circuit.library import TwoLocal

n_qubits = 11
qaoa_embedding_option2 = QuantumCircuit(n_qubits)


# create a list of strings with cyclic permutations of "ZZIIIIIIIII"
s = "I"*n_qubits
multiZ_strings = []
for i in range(n_qubits-1):

    z_str = list(s)
    z_str[i] = "Z"
    z_str[i+1] = "Z"
    multiZ_strings.append("".join(z_str))

multiZ_strings.append("Z"+"I"*(n_qubits-2)+"Z")


# Option1 : using qiskit qaoa ansatz : make QAOA embedding
cost_operator = SparsePauliOp(multiZ_strings)
# qaoa_embedding = qaoa_ansatz(cost_operator, reps=1, insert_barriers=True)

# Option2 : copying pennylabe qaoa embedding
# use qiskit.circuit.library.NLocal to create rotation + entanglement layers ( for ML, VQE  )
# for the first layer, use 'rx' so that it can be used for the input data embedding.
# for the remaining layers, all the parameters are trainable.
# use <your_cricuit>.parameters to get the list of parameters in the circuit

class QAOAEmbedding(QuantumCircuit):
    def __init__(self, n_qubits, reps=1):
        super().__init__(n_qubits)
        self.reps = reps

        self.theta_params_vector



qaoa_embedding_option2 = TwoLocal(n_qubits, 'rx', 'rzz', entanglement='circular', reps=1, insert_barriers=True, skip_final_rotation_layer=True)
qaoa_embedding_option2.barrier()

# add final rotation layer
final_rotation = QuantumCircuit(n_qubits)
theta_params_vector = ParameterVector('θ_1', n_qubits*2)
for i in range(n_qubits):
    final_rotation.ry(theta_params_vector[i], i)
for i in range(n_qubits):
    final_rotation.rx(theta_params_vector[i+n_qubits], i)
final_rotation.barrier()

qaoa_embedding_option2.compose(final_rotation, inplace=True)
qaoa_embedding_option2.decompose().draw('mpl', style='clifford')


# In[7]:


# create entangled layer
# create strong entanglement to get a quantum advantage

class strong_layer(QuantumCircuit):
    def __init__(self, n_qubits, reps=1):
        super().__init__(n_qubits)  # inherit n_qubits attribute from QuantumCircuit
        self.reps = reps

        # define the parameters that will be used in Ugate(a,b,c)
        self.params_a = ParameterVector('θ_a', n_qubits*reps)
        self.params_b = ParameterVector('θ_b', n_qubits*reps)
        self.params_c = ParameterVector('θ_c', n_qubits*reps)

        # repeat the following block reps times
        for rep_idx in range(reps):

            # apply U gate to each qubit
            for i in range(n_qubits):
                self.u(self.params_a[i+rep_idx*n_qubits], self.params_b[i+rep_idx*n_qubits], self.params_c[i+rep_idx*n_qubits], i)

            # apply Cnot layer in a circular way
            for i in range(n_qubits-1):
                self.cx(i, i+1)
            self.cx(n_qubits-1, 0)
            self.barrier()


        display(self.draw('mpl', style='clifford'))

# create layer instance

strong_entangled_layer = strong_layer(n_qubits=11, reps=2)





# In[8]:


# create ansatz circuit

ansatz = QuantumCircuit(n_qubits)
ansatz.compose(qaoa_embedding_option2, inplace=True)
ansatz.compose(strong_entangled_layer, inplace=True)
# display(ansatz.decompose().draw('mpl', style='clifford'))

ansatz = transpile(ansatz)
display(ansatz.decompose().draw('mpl', style='clifford'))



# In[9]:


# import GPU simulator in qiskit_aer simulator

from qiskit_aer import AerError

try:
    simulator_gpu = AerSimulator(method='statevector', device = 'GPU')

except AerError as e:
    print(e)


# In[10]:


# designate the simulator backend to the Estimator primitive

primitive_estv2 = EstimatorV2( options= { "backend_options" : {"method" : "statevector", "device" : "GPU"}})




# parity maps bitstrings to 0 or 1 ( depending on the parity of the number of '1' in bitstring)
def parity(x):
    return "{:b}".format(x).count("1") % 2

output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping.



# create a QNN instance with simulator.
estimator_qnn = EstimatorQNN(
    circuit=ansatz,
    estimator = primitive_estv2,
    input_gradients=True,
    input_params = ansatz.parameters[:11],  # input data embedding parameters : the first 11 parameters
    weight_params = ansatz.parameters[11:]  # trainable parameters : the rest of the parameters
)

# create a QNN instance with sampler.
# sampler_qnn = SamplerQNN(
#     circuit=ansatz,
#     sampler = SamplerV2(),
#     interpret = parity,
#     output_shape=output_shape,
#     input_gradients=True,
#     input_params = ansatz.parameters[:11],  # input data embedding parameters : the first 11 parameters
#     weight_params = ansatz.parameters[11:]  # trainable parameters : the rest of the parameters
# )


np.random.seed(0)
initial_weights = 0.1*(2* np.random.rand(99) -1 )
estimator_qnn.forward(X_train.cpu(), initial_weights )


# ### Using TorchConnector to make hybrid-quantum

# In[ ]:


# Create TorchConnector model

np.random.seed(0)
initial_weights = 0.1*(2* np.random.rand(99) -1 )


# create TorchConnector instance ( connects torch to qiskit )
qnn_estimator = TorchConnector(estimator_qnn, initial_weights=initial_weights).to(device)
# qnn_sampler = TorchConnector(sampler_qnn, initial_weights=initial_weights).to(device)

qnn = qnn_estimator

# create a Custom QNN model.
# For future reference, you can build hybrid model that combine classical MLP and quantum circuit.
class MyQNN4L3Muon(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = qnn

    def forward(self,x):
        return (self.qnn(x) + 1.0 ) / 2.0


model1 = MyQNN4L3Muon().to(device)

# test the model
print(f"Test the model w/ initial weights and 1st sample : {model1(X_train).size()}")


# In[ ]:


# define optimizer, scheduler, and loss function
optimizer = optim.Adam(model1.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# from tqdm.notebook import tqdm  # Notebook version for better display in Jupyter
from tqdm import tqdm  # Notebook version for better display in Jupyter

# Early stopping parameters
best_test_loss = float('inf')
patience_counter = 0
early_stopping_patience = 5

# good balance between MAE and MSE
loss_function = nn.MSELoss()

epochs = 100
batch_size = 20

# for tracking purpose
train_losses = []
test_losses = []
r2_score_value = []   # r2 = how well the model predicts the data

for epoch in range(epochs):

    # set the model to train mode; ensures gradients are computed
    # ensures dropout layers are active
    model1.train()

    # shuffle the data
    permutation = torch.randperm(len(X_train))
    running_loss = 0.0

    for i in range(0, len(X_train), batch_size):  # loop over mini-batch per one epoch
        print(f"{epoch} epoch has been proceeding,  {i+1} batch roop has been proceeding")
        indices = permutation[i:i+batch_size]  # 0-9th permuted indices, 10-19, 20-29, ... when batch_size=10
        X_batch, y_batch = X_train[indices], y_train[indices]  # get the mini-batch data

        print(f"X_batch = {X_batch.size()}, y_batch = {y_batch.size()}")

        # zero the parameter gradients : in default setting, the gradients from previous epoch are accumulated
        optimizer.zero_grad()

        # forward pass
        y_pred_estimator = model1(X_batch).squeeze()
        y_pred_sampler   = model1(X_batch)[:,0]   # choose 1st column as output class of sampler.

        # designate which primitives will be used
        y_pred = y_pred_estimator

        if y_pred.size() != y_batch.size():
            print(f"y_pred size: {y_pred.size()}, y_batch size: {y_batch.size()}")
            raise ValueError("y_pred and y_batch have different sizes")


        loss = loss_function(y_pred.to(device), y_batch.to(device))  # compute the loss

        loss.backward()  # backward pass for computing the gradients

        optimizer.step()  # update the weights
        running_loss += loss.item()  # accumulate the loss

    avg_train_loss = running_loss / ( len(X_train) / batch_size)
    train_losses.append(avg_train_loss)


    # set the model to evaluation mode; ensures gradients are not computed
    # ensures dropout layers are inactive
    model1.eval()
    with torch.no_grad(): # no need to compute the gradients
        y_test_pred = model1(X_test).squeeze()
        test_loss = loss_function(y_test_pred, y_test)
        test_losses.append(test_loss.item())

    scheduler.step(test_loss)

    # Compute metrics for logging
    y_test_prob = y_test_pred.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()
    y_pred_labels = (y_test_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test_np, y_pred_labels)
    
    # Log metrics without interfering with the progress bars
    tqdm.write(f"Epoch {epoch+1}/{epochs}")
    tqdm.write(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
    tqdm.write(f"Accuracy: {accuracy:.4f}")
    
    # Early stopping check
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
            break

    # convert the final prediction ( number between -1 and 1 ) to numpy array
    y_pred_final = y_test_pred.cpu().numpy()
    y_test_final = y_test.cpu().numpy()

    # compute the r2 score
    r2 = r2_score(y_test_final, y_pred_final)
    r2_score_value.append(r2)


    # print log
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"R2 Score: {r2:.4f}")

    for name, param in model1.named_parameters():
        print(f"{name} - Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}")

#############################################################################################################
# Training is done.

# Save the model's parameters

try:
    torch.save(model1.state_dict(), "./qnn_model.pth")
except Exception as e:
    print(e)

# Load the model later
# qnn_model.load_state_dict(torch.load("qnn_model.pth"))


# In[ ]:


# plot the loss and r2 score
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Test Loss")
plt.show()
plt.savefig("./loss_plot.png")

plt.figure(figsize=(6, 4))
plt.plot(r2_score_value, label="R² Score")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("Epoch-wise R² Score per Output Feature")
plt.legend()
plt.show()
plt.savefig("./r2_score_plot.png")


# After training, evaluate the model on the test data
with torch.no_grad():
    y_pred_final = model1(X_test).cpu().numpy()
    y_test_np = y_test.cpu().numpy()


# Plot the predicted probabilities against the true labels
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test_np)), y_pred_final, c=y_test_np, cmap='bwr', edgecolors='none', s=35)
plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold = 0.5')

# Add probability value on top of the points
# for i in range(len(y_test_np)):
#     plt.text(i, y_pred_final[i] + 0.05, f'{float(y_pred_final[i]):.2f}', ha='center', fontsize=12)

# Create a color legend manually
legend_labels = {0: "Class 0", 1: "Class 1"}
colors = ['blue', 'red']  # Colors matching 'bwr' colormap

for label, color in legend_labels.items():
    plt.scatter([], [], c=colors[label], label=color, s=40)

plt.legend(title="True Labels", loc="upper left", bbox_to_anchor=(1, 1))

step = 5
plt.xticks(range(0, len(y_test_np)+step, step), rotation = 45)
plt.xlabel("Sample Index")
plt.ylabel("Predicted Probability")
plt.title("Binary Classification Scatter Plot")
plt.ylim(0, 1.1)

plt.show()
plt.savefig("./scatter_plot.png")


# ### *Notice about choosing loss function for Quantum Classificaiton
# 
# In classical machine learning, the general rule of thumb is to apply a Cross-Entropy loss to classification tasks, and MSE loss to regression tasks. However, this recommendation is given under the assumption that the output of the classification network is a class probability value in the range (usually this is achieved through a Softmax layer). Because the following example for EstimatorQNN does not include such layer, and we don’t apply any mapping to the output (the following section shows an example of application of parity mapping with SamplerQNNs), the QNN’s output can take any value in the range . In case you were wondering, this is the reason why this particular example uses MSELoss for classification despite it not being the norm (but we encourage you to experiment with different loss functions and see how they can impact training results).
