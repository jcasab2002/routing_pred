# models_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt

# --- El modelo VAE en PyTorch ---
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_output_mean = nn.Linear(hidden_dim, 1)
        self.fc_output_log_var = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        return self.fc_mean(h), self.fc_log_var(h)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder_net(z)
        return self.fc_output_mean(h), self.fc_output_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction_mean, reconstruction_log_var = self.decode(z)
        return reconstruction_mean, reconstruction_log_var, mu, log_var

# --- Función de Pérdida del VAE (ELBO) ---
def vae_loss(y_true: torch.Tensor, y_pred_mean: torch.Tensor, y_pred_log_var: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Pérdida de Reconstrucción (Log-Likelihood de una Log-Normal)
    reconstruction_loss = 0.5 * torch.mean(
        torch.exp(-y_pred_log_var) * (y_true - y_pred_mean)**2 + y_pred_log_var
    )
    
    # Pérdida de Divergencia KL
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    kl_loss = torch.mean(kl_loss)
    
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss

# --- Clase de utilidad para entrenar y predecir ---
class TimeDurationVAEPytorch:
    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dim: int = 128, device: str = 'cpu'):
        self.device = device
        self.model = VAE(input_dim, latent_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, epochs: int = 50, batch_size: int = 32):
        print("Entrenando VAE en PyTorch...")
        self.model.train() # Pone el modelo en modo entrenamiento
        
        # Convertir datos de pandas a tensores de PyTorch
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                reconstruction_mean, reconstruction_log_var, mu, log_var = self.model(data)
                
                loss, recon_loss, kl_loss = vae_loss(target, reconstruction_mean, reconstruction_log_var, mu, log_var)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(dataloader):.4f}")
        print("Entrenamiento del VAE en PyTorch completado.")

    def predict(self, new_data: pd.DataFrame, n_samples: int = 100) -> pd.Series:
        self.model.eval() # Pone el modelo en modo evaluación
        
        X_tensor = torch.tensor(new_data.values, dtype=torch.float32).to(self.device)
        
        with torch.no_grad(): # Desactiva el cálculo de gradientes
            mu_pred, log_var_pred = self.model.encode(X_tensor)
        
        predicted_means = []
        for i in range(len(new_data)):
            current_mu = mu_pred[i]
            current_log_var = log_var_pred[i]

            dist = Normal(current_mu, torch.exp(0.5 * current_log_var))
            latent_samples = dist.sample((n_samples,))
            
            output_mean, _ = self.model.decode(latent_samples)
            
            predicted_means.append(torch.mean(torch.exp(output_mean)).item())
            
        return pd.Series(predicted_means, index=new_data.index)
    def plot_predictions_vs_actual(self, X_test: pd.DataFrame, y_test: pd.Series):
        
        predictions = self.predict(X_test, n_samples=100)
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.scatter(y_test, predictions, alpha=0.5)
        max_val = max(y_test.max(), predictions.max())
        ax1.plot([0, max_val], [0, max_val], 'r--')
        ax1.set_title('Predicciones vs. Valores Reales')
        ax1.set_xlabel('Tiempo de Parada Real')
        ax1.set_ylabel('Predicción del VAE (Media)')
        ax1.grid(True)
        
        ax2 = plt.subplot(1, 2, 2)
        residuals = y_test - predictions
        ax2.hist(residuals, bins=50, edgecolor='k')
        ax2.set_title('Histograma de Residuos')
        ax2.set_xlabel('Error de Predicción (Real - Predicción)')
        ax2.set_ylabel('Frecuencia')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_learned_distributions(self, X_sample: pd.DataFrame):
        self.model.eval()
        X_tensor = torch.tensor(X_sample.values, dtype = torch.float32).to(self.device)

        with torch.no_grad():
            mu_pred, log_var_pred = self.model.encode(X_tensor)
        
        fig, axes = plt.subplots(nrows=1, ncols=len(X_sample), figsize=(len(X_sample) * 4, 5))
        if len(X_sample) == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            current_mu = mu_pred[i]
            current_log_var = log_var_pred[i]
            dist = Normal(current_mu, torch.exp(0.5 * current_log_var))

            latent_samples = dist.sample((1000, ))
            output_mean, _ = self.model.decode(latent_samples)
            predicted_durations = torch.exp(output_mean).detach().cpu().numpy().flatten()

            ax.hist(predicted_durations, bins=50, density=True, alpha=0.6, label='Distribucion predicha')
            ax.set_title(f"Distribucion vae ejemplo {i+1}")
            ax.set_xlabel('tiempo de parada predicho')
            ax.set_ylabel('densidad')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()