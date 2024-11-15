from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# EnhancedModel sınıfını ve modeli yükleme
class EnhancedModel(nn.Module):
    def __init__(self):
        super(EnhancedModel, self).__init__()
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

model = EnhancedModel()
model.load_state_dict(torch.load('enhanced_model.pth', weights_only=True))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = torch.tensor(data['input']).float()
    with torch.no_grad():
        prediction = model(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
