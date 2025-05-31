import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(model, optimizer, loss_fn, X_train, y_train, epochs=30, batch_size=32):
   
    # Creazione del DataLoader per i dati di addestramento
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    os.makedirs("./model_folder", exist_ok=True)
    # Ciclo per ogni epoca
    for epoch in range(epochs):
        model.train(True)  # Imposta il modello in modalità di addestramento
        running_loss = 0.0

        # Ciclo sui batch di dati
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Azzeramento dei gradienti

            outputs = model(inputs)  # Passaggio in avanti
            loss = loss_fn(outputs, labels)  # Calcolo della perdita
            loss.backward()  # Calcolo dei gradienti
            optimizer.step()  # Aggiornamento dei pesi

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model_path = os.path.join("./model_folder", "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model Saved in {model_path}")
    return model