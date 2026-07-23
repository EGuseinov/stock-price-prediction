"""
LSTM ve GRU Model Tanimlari
============================
Amazon (AMZN) hisse senedi fiyat tahmini icin kullanilan
PyTorch tabanli zaman serisi modelleri.

Kullanim:
    from models import LSTM, GRU, TunedLSTM, TunedGRU
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM (Long Short-Term Memory) modeli.

    3 kapi (forget, input, output) ve bir cell state kullanarak
    uzun vadeli bagimliliklari ogrenir.

    Parametreler:
        input_dim  (int): Giris boyutu (1 = sadece kapanis fiyati)
        hidden_dim (int): Gizli katman boyutu
        num_layers (int): LSTM katman sayisi
        output_dim (int): Cikis boyutu (1 = tek fiyat tahmini)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    """
    GRU (Gated Recurrent Unit) modeli.

    LSTM'in basitlestirilmis hali. 2 kapi (reset, update) kullanir.
    Cell state yoktur, sadece hidden state vardir.
    Daha az parametre = daha hizli egitim.

    Parametreler:
        input_dim  (int): Giris boyutu
        hidden_dim (int): Gizli katman boyutu
        num_layers (int): GRU katman sayisi
        output_dim (int): Cikis boyutu
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class TunedLSTM(nn.Module):
    """
    Dropout ekli LSTM modeli (hiperparametre optimizasyonu icin).

    Parametreler:
        input_dim    (int): Giris boyutu
        hidden_dim   (int): Gizli katman boyutu
        num_layers   (int): LSTM katman sayisi
        output_dim   (int): Cikis boyutu
        dropout_rate (float): Dropout orani (0.0 - 1.0)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.0):
        super(TunedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class TunedGRU(nn.Module):
    """
    Dropout ekli GRU modeli (hiperparametre optimizasyonu icin).

    Parametreler:
        input_dim    (int): Giris boyutu
        hidden_dim   (int): Gizli katman boyutu
        num_layers   (int): GRU katman sayisi
        output_dim   (int): Cikis boyutu
        dropout_rate (float): Dropout orani (0.0 - 1.0)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.0):
        super(TunedGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
