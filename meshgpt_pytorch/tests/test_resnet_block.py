import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import pytest
from meshgpt_pytorch.blocks import ResnetBlock

@pytest.mark.parametrize("dim, dim_out, groups, dropout", [
    (64, 64, 8, 0.1),
])
def test_resnet_block_training_with_mlflow(dim, dim_out, groups, dropout):
    # Initialize MLFlow
    mlflow.set_tracking_uri("http://localhost:5001")

    mlflow.set_experiment("ResnetBlock Training Test")
    
    with mlflow.start_run():
        # Initialize a ResnetBlock
        resnet_block = ResnetBlock(dim=dim, dim_out=dim_out, groups=groups, dropout=dropout)
        
        # Log model parameters
        mlflow.log_param("dim", dim)
        mlflow.log_param("dim_out", dim_out)
        mlflow.log_param("groups", groups)
        mlflow.log_param("dropout", dropout)
        
        # Create a simple dataset
        x = torch.randn(100, dim, 50)  # (batch_size, channels, length)
        y = torch.randn(100, dim_out, 50)  # Target is the same shape
        
        # Define a simple mean squared error loss
        criterion = nn.MSELoss()
        
        # Use an optimizer
        optimizer = optim.Adam(resnet_block.parameters(), lr=0.001)
        
        # Training loop
        initial_loss = None
        for epoch in range(10):  # Run for a few epochs
            optimizer.zero_grad()
            
            # Forward pass
            output = resnet_block(x)
            
            # Compute loss
            loss = criterion(output, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Record the initial loss
            if initial_loss is None:
                initial_loss = loss.item()
            
            # Log the loss
            mlflow.log_metric("loss", loss.item(), step=epoch)
            
            # Print loss for debugging
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        # Check if the loss has decreased
        assert loss.item() < initial_loss, "Loss did not decrease, training might not be working properly"

if __name__ == "__main__":
    test_resnet_block_training_with_mlflow()
    print("Training test with MLFlow passed!")