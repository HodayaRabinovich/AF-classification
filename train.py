import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

criterion = nn.MSELoss()


class ConvUnit(nn.Module):
    def __init__(self, unit):
        super(ConvUnit, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.add(x, residual)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_units = nn.ModuleList([ConvUnit(unit) for unit in range(5)])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 1) # 1 output for regression
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        for unit in self.conv_units:
            x = unit(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, name):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # arrange data
    train_dataset = TensorDataset(x_train.unsqueeze(1),
                                  y_train.unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(x_valid.unsqueeze(1),
                                  y_valid.unsqueeze(1))
    valid_loader = DataLoader(valid_dataset, batch_size=x_valid.size(0), shuffle=True)

    train_loss = np.zeros(epochs)
    val_loss_arr = np.zeros(epochs)
    lr = np.zeros(epochs)

    print("start training")
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = 0
        for inputs, labels in valid_loader:
            # Forward pass
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
        val_loss /= len(valid_loader)
        # Update learning rate scheduler
        lr_scheduler.step(val_loss)

        # Print current learning rate
        print("Epoch:", epoch, 'Loss:', loss.detach(), "val loss", val_loss, "Learning Rate:",  optimizer.param_groups[0]['lr'])
        train_loss[epoch] = loss.detach()
        val_loss_arr[epoch] = val_loss
        lr[epoch] = optimizer.param_groups[0]['lr']

    model_name = name + '.pth'
    torch.save(model.state_dict(), model_name)
    return train_loss, val_loss_arr, lr

def test_model(model, x_test, y_test, name):
    # arrange data
    test_dataset = TensorDataset(x_test.unsqueeze(1),
                                  y_test.unsqueeze(1))
    test_loader = DataLoader(test_dataset, x_test.size(0), shuffle=False)

    model_name = name + '.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()
    test_loss = 0
    pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            pred.append(outputs)
        test_loss /= len(test_loader)

    print("test loss: " + str(test_loss))

    return np.array(pred[0])



    '''
def conv_unit(unit, input_layer):
    s = '_' + str(unit)
    # layer = nn.Conv1d(name='Conv1' + s, filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(
    #     input_layer)
    # layer = keras.Conv1D(name='Conv2' + s, filters=32, kernel_size=5, strides=1, padding='same', activation=None)(layer)
    # layer = keras.Add(name='ResidualSum' + s)([layer, input_layer])
    # layer = keras.Activation("relu", name='Act' + s)(layer)
    # layer = keras.MaxPooling1D(name='MaxPool' + s, pool_size=5, strides=2)(layer)
    return layer


def cnn_model(input_layer, mode, params):
    current_layer = keras.Conv1D(filters=32, kernel_size=5, strides=1)(input_layer)

    for i in range(5):
        current_layer = conv_unit(i + 1, current_layer)

    current_layer = keras.Flatten()(current_layer)
    current_layer = keras.Dense(32, name='FC1', activation='relu')(current_layer)
    logits = keras.Dense(5, name='Output')(current_layer)

    print('Parameter count:', parameter_count())
    return logits
'''