import torch

def compute_spectral_stats(tensor):
    """Computes spectral statistics of a 2D tensor on CPU to save VRAM."""
    with torch.no_grad():
        if tensor.ndim < 2:
            return {"max": 0.0, "mean": 0.0, "gap": 0.0, "stable_rank": 0.0}
        
        # Flatten to 2D if needed (e.g. for conv or multiple heads)
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        
        # We need at least 2 singular values for gap
        if t.size(0) < 2 or t.size(1) < 2:
            s = torch.linalg.svdvals(t)
            spectral_max = s[0].item() if len(s) > 0 else 0.0
            return {
                "max": spectral_max,
                "mean": s.mean().item() if len(s) > 0 else 0.0,
                "gap": 1.0,
                "stable_rank": 1.0
            }

        s = torch.linalg.svdvals(t)
        
        spectral_max = s[0].item()
        spectral_mean = s.mean().item()
        spectral_gap = (s[0] / s[1]).item() if len(s) > 1 else 1.0
        stable_rank = (torch.norm(t, p='fro')**2 / (s[0]**2)).item() if spectral_max > 0 else 1.0
        
        return {
            "max": spectral_max,
            "mean": spectral_mean,
            "gap": spectral_gap,
            "stable_rank": stable_rank
        }

def compute_singular_values(tensor, n=10):
    """Returns top n singular values (computed on CPU)."""
    with torch.no_grad():
        if tensor.ndim < 2:
            return [0.0] * n
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        s = torch.linalg.svdvals(t)
        vals = s[:n].tolist()
        if len(vals) < n:
            vals += [0.0] * (n - len(vals))
        return vals
