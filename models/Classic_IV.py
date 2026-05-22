import torch
from torch import nn

class ClassicIV(nn.Module):
    def __init__(self, n_fft=800, hop_length=320, eps=1e-8):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)

    def forward(self, raw_wav):
        origin_dtype = raw_wav.dtype
        B, C, T = raw_wav.shape
        raw_wav_flat = raw_wav.view(-1, T).to(torch.float32)
        window = self.window.to(raw_wav_flat.device).to(torch.float32)
        stft = torch.stft(
            raw_wav_flat, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window, 
            return_complex=True,
            center=True 
        )
        
        _, F_bins, T_frames = stft.shape
        stft = stft.view(B, C, F_bins, T_frames)
        
        W = stft[:, 0:1, :, :]
        XYZ = stft[:, 1:, :, :]
        
        I = torch.real(torch.conj(W) * XYZ)
        
        mag_sq_W = torch.abs(W)**2
        mag_sq_XYZ = torch.abs(XYZ)**2
        E = self.eps + mag_sq_W + (mag_sq_XYZ.sum(dim=1, keepdim=True) / 3.0)
        
        I_norm = I / E
        
        # [B, 3, F, T] -> [B, T, 3*F]
        I_norm = I_norm.permute(0, 3, 1, 2).contiguous()
        y = I_norm.view(B, T_frames, -1).to(origin_dtype)
        return y
    
if __name__ == "__main__":
    model = ClassicIV().cuda()

    signal = torch.randn([4, 4, 160000]).cuda()
    with torch.no_grad():
        output = model(signal)
        print(output.shape)