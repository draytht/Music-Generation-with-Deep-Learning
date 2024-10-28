import torch
import numpy as np

def train(model, data, criterion, optimizer, num_epochs=10, batch_size=32):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Shuffle data for each epoch
        np.random.shuffle(data)
        
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i:i + batch_size]
            batch = torch.tensor(batch, dtype=torch.float32)
            
            # Split data into inputs and targets
            inputs = batch[:, :-1, :]  # All except last time step
            targets = batch[:, -1, :]  # Predict the last time step
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            batch_count += 1
        
        # Print average loss per epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / batch_count:.4f}")
