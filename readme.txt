# StatSense

## Machine learning model tracking simplified

* Get started
```code
pip install -r requirements.txt
```

* Write model.py file with tracker object from TorchTrack for logging as:
1. For epoch logging:
```python
tracker.log_epoch(
            loss=loss.item(), 
            accuracy=r2  # Using R2 score as the performance metric
        )
```

2. For final logging:
```python
tracker.log(
        hyperparameters=hyperparams, 
        metrics={'r2_score': best_r2},
        model_type = "Regression",
        model_data = """
        # Model class
        class DiabetesNet(nn.Module):
            def __init__(self, input_size, hidden_layers):
                super(DiabetesNet, self).__init__()
                layers = []
                prev_size = input_size
                
                for size in hidden_layers:
                    layers.append(nn.Linear(prev_size, size))
                    layers.append(nn.ReLU())
                    prev_size = size
                
                layers.append(nn.Linear(prev_size, 1))
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)

        # Instantiate model
        model = DiabetesNet(X_train.shape[1], hyperparams['hidden_layers']).to(device)

        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        """
    )
```

* Finally run streamlit app to see the results and AI analysis:
```code
streamlit run app.py
```