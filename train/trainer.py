import torch
from torch.optim import Adam

def train_vae(model, train_loader, test_loader, epochs=20, lr=1e-4, patience=3, device='cuda'):
    optimiser = Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_total, train_kl, train_recon = 0, 0, 0
        kl_weight = 0.6 * (1 - torch.exp(torch.tensor(-epoch / 5.0)))

        for batch in train_loader:
            img = batch[0].to(device)
            optimiser.zero_grad()
            recon, mu, logvar = model(img)
            kl, recon_loss, total_loss = model.loss(img, recon, mu, logvar, kl_weight=kl_weight)
            total_loss.backward()
            optimiser.step()

            train_total += total_loss.item()
            train_kl += kl.item()
            train_recon += recon_loss.item()

        # Validation
        model.eval()
        val_total, val_kl, val_recon = 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                img = batch[0].to(device)
                recon, mu, logvar = model(img)
                kl, recon_loss, total_loss = model.loss(img, recon, mu, logvar)

                val_total += total_loss.item()
                val_kl += kl.item()
                val_recon += recon_loss.item()

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_total/len(train_loader):.4f}")
        print(f"Val Loss: {val_total/len(test_loader):.4f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break
    
    return model, optimiser

def save_model(model, optimiser, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict()
    }, save_path)
    print(f"Model saved at {save_path}")
