import torch
import torch.nn as nn


class ManyToOneLSTM(nn.Module):

    def __init__(self, encoding_size, output_size):  
        super(ManyToOneLSTM, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, output_size)  # Use output_size for the final layer (5 possible emojis)

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def forward(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1])  # Take the output of the last time step


char_encodings = {
    ' ': [1., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' (space)
    'h': [0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    'a': [0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'a'
    't': [0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 't'
    'r': [0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'r'
    'c': [0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'c'
    'f': [0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'f'
    'l': [0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'l'
    'p': [0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 'p'
    's': [0., 0., 0., 0., 0., 0., 0., 0., 1.]   # 's'
}

emoji_encodings = {
    'hat': 0,  # 🎩
    'rat': 1,  # 🐀
    'cat': 2,  # 🐈
    'flat': 3,  # 🏢
    'cap': 4,  # 🧢
}

index_to_emoji = ['🎩', '🐀', '🐈', '🏢', '🧢']

# Pad the words to the maximum length so that each word is the same length
def pad_word(word, max_length):
    return word + ' ' * (max_length - len(word))

max_length = 4  

x_train = []
y_train = []

for word, emoji_idx in emoji_encodings.items():
    padded_word = pad_word(word, max_length)
    x_train.append([char_encodings[char] for char in padded_word])
    y_train.append(emoji_idx)

x_train = torch.tensor(x_train)  
y_train = torch.tensor(y_train)  

# Transpose x_train to shape (max_length, batch_size, encoding_size)
x_train = x_train.permute(1, 0, 2)

# Define the model with the correct output size
model = ManyToOneLSTM(encoding_size=len(char_encodings[' ']), output_size=len(emoji_encodings))

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
batch_size = x_train.shape[1]  # Get the batch size dynamically from the input
for epoch in range(500):
    model.reset(batch_size)
    output = model(x_train)
    loss = loss_fn(output, y_train)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

test_words = ['at', 'rats']
for word in test_words:
    padded_word = pad_word(word, max_length)
    x_test = torch.tensor([[char_encodings[char] for char in padded_word]])
    x_test = x_test.permute(1, 0, 2)  
    model.reset(batch_size=1)
    output = model(x_test)
    predicted_emoji_idx = torch.argmax(output).item()
    print(f'Word: {word}, Predicted emoji: {index_to_emoji[predicted_emoji_idx]}')
