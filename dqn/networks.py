import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN(nn.Module):
    def __init__(self, action_size, hidden_size, input_shape=(1, 4, 11, 11), filter_size=64):
        super(DQN_CNN, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(4, filter_size // 4,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(
            filter_size // 4, filter_size // 2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(filter_size // 2, filter_size,
                               kernel_size=3, stride=1)

        def conv_output_size(size, kernel_size=3, stride=1):
            return (size - kernel_size) // stride + 1

        # Calculate the output size after each conv layer
        conv_w = conv_output_size(conv_output_size(
            conv_output_size(self.input_shape[2])))
        conv_h = conv_output_size(conv_output_size(
            conv_output_size(self.input_shape[3])))
        conv_output_size = filter_size * conv_w * conv_h

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQN_LSTM(nn.Module):
    def __init__(self, action_size, hidden_size, input_shape=(1, 4, 11, 11)):
        super(DQN_LSTM, self).__init__()
        self.input_shape = input_shape

        # Flatten the spatial dimensions for RNN processing
        self.rnn_input_size = input_shape[1] * input_shape[2]
        self.sequence_length = input_shape[3]

        # Deep RNN layer
        self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=hidden_size,
                           num_layers=2, batch_first=True, dropout=0.5)

        # Additional Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, -1)

        x, _ = self.rnn(x)
        x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
