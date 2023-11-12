import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

def train(model, train_dataset, val_dataset, device, 
        batch_size, epochs, lr, patience, plot=True):
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    counter = 0

    # Train the model
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['intent'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1).cpu()
            labels = labels.cpu()

            train_loss += loss.item()
            train_acc += accuracy_score(labels, preds)

            avg_train_loss = train_loss/(len(train_loader.dataset)/batch_size)
            avg_train_acc = train_acc/(len(train_loader.dataset)/batch_size)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)


        # Evaluate the model on the val set
        model.eval()
        val_loss = 0
        val_acc = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['intent'].to(device)

                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1).cpu()
                val_acc += accuracy_score(labels.cpu(), preds)

                val_preds.extend(preds)
                val_labels.extend(labels.cpu())

                avg_val_loss = val_loss/(len(val_loader.dataset)/batch_size)
                avg_val_acc = val_acc/(len(val_loader.dataset)/batch_size)

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        
        # Check if early stopping conditions are met
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping triggered.')
                epochs = epoch +1
                break

    if plot:
        # Print classification report
        print(classification_report(val_labels, val_preds))

        # Plot training and validation metrics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), train_losses, label='Train Loss')
        plt.plot(range(epochs), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
        plt.plot(range(epochs), val_accuracies, label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()