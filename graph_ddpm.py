import torch, imageio
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as tnn
import sklearn.preprocessing as skp
from torch_geometric.data import DataLoader
from numpy.core.function_base import linspace
import pygsp as gsp
import numpy as np
from pygsp import graphs, filters, plotting
from graph_gen_dataset import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def get_adj_matrix(data):
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.

    return adj_matrix


def graph_laplacian(adj):
    # Check if the matrix is symmetric or not, and make it symmetric
    # G = graphs.Graph(adj)

    # if G.is_directed():
    t = np.transpose(adj)
    adj = adj + t

    # Generate graph laplacian U
    D = np.diag(np.sum(adj, axis=1))
    D_inv = np.linalg.inv(D)
    U = D @ adj @ D_inv

    return U


def get_significant_k_basis(U, lamda):
    # Find the eigenvalues and eigenvectors in U
    eigenvalues, eigenvectors = np.linalg.eig(U)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)

    # Normalize the eigenvalues using the formula
    sum_eigenvalues = np.sum(eigenvalues)
    normalized_eigenvalues = eigenvalues / sum_eigenvalues

    # Select the top-k value, if the values of eigenvalues sum up to lamda
    sorted_eigenvalues = np.sort(normalized_eigenvalues)[::-1]

    # Compute the cumulative sum of the eigenvalues
    cumulative_sum = np.cumsum(sorted_eigenvalues)

    # Find the index of the last eigenvalue whose cumulative sum is less than or equal to lamda
    last_index = np.where(cumulative_sum <= lamda)[0][-1]

    return last_index + 1


class graph_ddpm(nn.Module):
    def __init__(self, input_dim, inter_1, inter_2, inter_3, inter_4, kernel_0, kernel_1, fea_dim, time_dim):
        super(graph_ddpm, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(input_dim, inter_1, kernel_size=kernel_0, stride=1)
        self.conv2 = nn.Conv2d(inter_1, inter_2, kernel_size=kernel_0, stride=1)
        self.conv3 = nn.Conv2d(inter_2, inter_3, kernel_size=kernel_0, stride=1)
        self.conv4 = nn.Conv2d(inter_3, inter_4, kernel_size=kernel_0, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(fea_dim, 256)
        self.fc2 = nn.Linear(256, fea_dim)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1, inter_3, kernel_size=kernel_1, stride=2)
        self.upconv2 = nn.ConvTranspose2d(inter_3, inter_2, kernel_size=kernel_1, stride=2)
        self.upconv3 = nn.ConvTranspose2d(inter_2, inter_1, kernel_size=kernel_1 + 1, stride=1)
        self.upconv4 = nn.ConvTranspose2d(inter_1, input_dim, kernel_size=3, stride=1)

        # Activation function
        self.activation1 = nn.ReLU()
        self.activation2 = nn.LeakyReLU()

    def Graph_autoencoder_forward(self, x, z):
        # Obtain basic properties of graph and transform feature matrix to have the same shape as adjacency matrix
        node = z.shape[0]

        feature = z.shape[1]

        z = self.activation1(self.fc1(z))

        # Reshape the input matrix to be a 4D tensor
        x = torch.stack([x, z], dim=0)

        # Encoder (adjacency matrix)
        x = self.activation1(self.conv1(x))
        x = self.pool(x)

        x = self.activation1(self.conv2(x))

        x = self.activation1(self.conv3(x))
        x = self.pool(x)

        x = self.activation1(self.conv4(x))

        x = torch.mean(x, dim=0)

        # Decoder (fused matrix)
        x = x.view(-1, x.shape[0], x.shape[1])

        x = self.activation1(self.upconv1(x))

        x = self.activation2(self.upconv2(x))

        x = self.activation1(self.upconv3(x))

        x = self.upconv4(x)

        # Reshape the tensor back to the size of adjacency matrix and node feature matrix
        x1, x2 = torch.split(x, split_size_or_sections=1, dim=0)

        # squeeze the first dimension of x1 and x2

        x2 = torch.squeeze(x2, dim=0)
        x2 = self.fc2(x2)

        return x1, x2

    def forward(self, x, z):
        return self.Graph_autoencoder_forward(x, z)


def generate_noise_matrices(num_matrices):
    matrices = []

    for _ in range(num_matrices):
        matrix = torch.rand(256, 256)
        matrices.append(torch.Tensor(matrix))

    return matrices


# define model

model = graph_ddpm(input_dim=2, inter_1=4, inter_2=8, inter_3=16, inter_4=32, kernel_0=3, kernel_1=6,
                             fea_dim=5, time_dim=8)

num_epochs, learning_rate = 100, 1e-4

# specify the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# move model to GPU
model = model.to(device)

# make dataloader
all_dates, comlist = find_common_days(2015, 2015)
method0, method1 = 'heat', 'threshold'
train_dataset = MyDataset(len(all_dates), all_dates, comlist, method0, method1,
                          root='/content/drive/MyDrive/Zinuo_Project/2_Graph_Generation/Generated_Dataset_2015')
loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# training process
model.train()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# init

noise = generate_noise_matrices(len(train_dataset))
noise = torch.nn.utils.rnn.pad_sequence(noise, batch_first=True, padding_value=0).to(device)

# Create a list to store the visualization artists
figures = []

for epoch in range(num_epochs):
    for i, graph in enumerate(train_dataset):
        if i == len(train_dataset) - 1:
            break

        x = graph.x.to(device)
        x_tar = train_dataset[i].x.to(device)

        adj_tar = train_dataset[i].edge_attr.to(device)

        recon_data, recon_feature = model(noise[i], x)

        loss = criterion(recon_data.squeeze(0), adj_tar) + criterion(recon_feature, x_tar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert the adjacency matrix to a numpy array
        if i == 0:
            adj_array = recon_feature.reshape(256, 5).cpu().detach().numpy()

            # Generate a heat map visualization of the adjacency matrix
            fig = plt.figure()
            plt.imshow(adj_array, cmap='hot', interpolation='nearest')
            plt.xlabel('Node Index')
            plt.ylabel('Node Index')
            plt.title('Graph Heat Map')
            plt.colorbar()

            # Save the figure as an image
            filename = f'frame_{epoch}_{i}.png'
            plt.savefig(filename)
            plt.close(fig)  # Close the figure to free memory

            # Append the image filename to the list
            figures.append(filename)

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Convert the saved images to a GIF file
images = []
for filename in figures:
    images.append(imageio.imread(filename))
imageio.mimsave('animation.gif', images, fps=1)

# Display the animation
plt.imshow(images[-1])  # Display the last frame as an example
plt.axis('off')
plt.show()

# test dataset
all_dates, comlist = find_common_days(2016, 2016)
method0, method1 = 'heat', 'threshold'
test_dataset = MyDataset(len(all_dates), all_dates, comlist, method0, method1,
                         root='/content/drive/MyDrive/Zinuo_Project/2_Graph_Generation/Generated_Dataset_2016')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# testing process
model.eval()
for epoch in range(num_epochs):
    for i, graph in enumerate(test_dataset):

        if i == len(test_dataset) - 1:
            break

        x = graph.x.to(device)

        x_tar = test_dataset[i + 1].x.to(device)

        adj = graph.edge_attr.to(device)

        adj_tar = test_dataset[i + 1].edge_attr.to(device)

        recon_data, recon_feature = model(adj, x)

        loss = criterion(recon_data.squeeze(0), adj_tar) + criterion(recon_feature, x_tar)

    # if epoch % 10 == 0:
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))