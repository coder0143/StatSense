import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from TorchTrack import TorchTrack

tracker = TorchTrack()
tracker.clean_previous_data()

class BreastCancerNet(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(BreastCancerNet, self).__init__()
        layers = []
        prev_size = input_size
        
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def prepare_data():
    # Load breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, hyperparams):
    # Clean previous data
    tracker = TorchTrack(experiment_name=f"Breast_Cancer_Run_{hyperparams['run']}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(hyperparams['seed'])
    np.random.seed(hyperparams['seed'])
    
    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Instantiate model
    model = BreastCancerNet(X_train.shape[1], hyperparams['hidden_layers']).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # Training loop
    best_accuracy = 0
    for epoch in range(hyperparams['epochs']):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            best_accuracy = max(best_accuracy, accuracy.item())
        
        # Log epoch data
        tracker.log_epoch(
            loss=loss.item(), 
            accuracy=accuracy.item()
        )
    
    # Log final results
    tracker.log(
        hyperparameters=hyperparams, 
        metrics={'accuracy': best_accuracy},
        model_type="Classification",
        model_data="""
        # Model class:
        class BreastCancerNet(nn.Module):
            def __init__(self, input_size, hidden_layers):
                super(BreastCancerNet, self).__init__()
                layers = []
                prev_size = input_size
                
                for size in hidden_layers:
                    layers.append(nn.Linear(prev_size, size))
                    layers.append(nn.ReLU())
                    prev_size = size
                
                layers.append(nn.Linear(prev_size, 1))
                layers.append(nn.Sigmoid())
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)

    # Instantiate model
    model = BreastCancerNet(X_train.shape[1], hyperparams['hidden_layers']).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        """
    )
    
    return best_accuracy

def main():
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data()
    
    # Different hyperparameter configurations
    hyperparameter_sets = [
        {
            'run': 1,
            'seed': 42,
            'hidden_layers': [20, 10],
            'learning_rate': 0.01,
            'epochs': 100
        },
        {
            'run': 2,
            'seed': 123,
            'hidden_layers': [30, 15],
            'learning_rate': 0.005,
            'epochs': 150
        },
        {
            'run': 3,
            'seed': 456,
            'hidden_layers': [40, 20, 10],
            'learning_rate': 0.001,
            'epochs': 200
        }
    ]
    
    # Run experiments with different configs
    for config in hyperparameter_sets:
        train_model(X_train, y_train, X_test, y_test, config)

if __name__ == '__main__':
    main()

# # Regression

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
# from TorchTrack import TorchTrack

# tracker = TorchTrack()
# tracker.clean_previous_data()

# class DiabetesNet(nn.Module):
#     def __init__(self, input_size, hidden_layers):
#         super(DiabetesNet, self).__init__()
#         layers = []
#         prev_size = input_size
        
#         for size in hidden_layers:
#             layers.append(nn.Linear(prev_size, size))
#             layers.append(nn.ReLU())
#             prev_size = size
        
#         layers.append(nn.Linear(prev_size, 1))
        
#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.model(x)

# def prepare_data():
#     # Load diabetes dataset
#     X, y = load_diabetes(return_X_y=True)
    
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Scale the features
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     # Convert to PyTorch tensors
#     X_train = torch.FloatTensor(X_train)
#     y_train = torch.FloatTensor(y_train).unsqueeze(1)
#     X_test = torch.FloatTensor(X_test)
#     y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
#     return X_train, y_train, X_test, y_test

# def train_model(X_train, y_train, X_test, y_test, hyperparams):
#     # Clean previous data
#     tracker = TorchTrack(experiment_name=f"Diabetes_Prediction_Run_{hyperparams['run']}")
    
#     # Set random seeds for reproducibility
#     torch.manual_seed(hyperparams['seed'])
#     np.random.seed(hyperparams['seed'])
    
#     # Move data to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     X_train, y_train = X_train.to(device), y_train.to(device)
#     X_test, y_test = X_test.to(device), y_test.to(device)
    
#     # Instantiate model
#     model = DiabetesNet(X_train.shape[1], hyperparams['hidden_layers']).to(device)
    
#     # Loss and Optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
#     # Training loop
#     best_r2 = float('-inf')
#     for epoch in range(hyperparams['epochs']):
#         # Forward pass
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Compute R2 Score
#         with torch.no_grad():
#             test_outputs = model(X_test)
#             # Convert to numpy for sklearn metrics
#             y_test_np = y_test.cpu().numpy()
#             test_outputs_np = test_outputs.cpu().numpy()
            
#             r2 = r2_score(y_test_np, test_outputs_np)
#             mse = mean_squared_error(y_test_np, test_outputs_np)
#             best_r2 = max(best_r2, r2)
        
#         # Log epoch data
#         tracker.log_epoch(
#             loss=loss.item(), 
#             accuracy=r2  # Using R2 score as the performance metric
#         )
    
#     # Log final results
#     tracker.log(
#         hyperparameters=hyperparams, 
#         metrics={'r2_score': best_r2}
#     )
    
#     return best_r2

# def main():
#     # Prepare data
#     X_train, y_train, X_test, y_test = prepare_data()
    
#     # Different hyperparameter configurations
#     hyperparameter_sets = [
#         {
#             'run': 1,
#             'seed': 42,
#             'hidden_layers': [20, 5],
#             'learning_rate': 0.01,
#             'epochs': 250
#         },
#         {
#             'run': 2,
#             'seed': 123,
#             'hidden_layers': [30, 15],
#             'learning_rate': 0.03,
#             'epochs': 250
#         },
#         {
#             'run': 3,
#             'seed': 456,
#             'hidden_layers': [40, 20, 10],
#             'learning_rate': 0.03,
#             'epochs': 250
#         }
#     ]
    
#     # Run experiments with different configs
#     for config in hyperparameter_sets:
#         train_model(X_train, y_train, X_test, y_test, config)

# if __name__ == '__main__':
#     main()