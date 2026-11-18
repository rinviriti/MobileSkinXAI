import torch,math
import torch.nn.functional as F

def js_divergence(p, q, epsilon=1e-8):
    p, q = p.float().clamp_min(epsilon), q.float().clamp_min(epsilon)
    mid = (p + q) / 2
    return .5 * ((p * (p.log() - mid.log())).sum(-1) + (q * (q.log() - mid.log())).sum(-1))

def m5_loss(sw, ss, tw, ts, target, weights, cfg):
    """Teacher logits are detached; all weights/gates use training labels only."""
    c = cfg['loss']
    eps, temperature = c['epsilon'], c['temperature']
    sw, ss, tw, ts = [value.float() for value in [sw, ss, tw.detach(), ts.detach()]]
    p, q = tw.softmax(-1), ts.softmax(-1)
    entropy = -.5 * ((p * p.clamp_min(eps).log()).sum(-1) + (q * q.clamp_min(eps).log()).sum(-1)) / math.log(4)
    divergence = js_divergence(p, q, eps)
    confident = torch.minimum(p.max(-1).values, q.max(-1).values) >= c['confidence_threshold']
    gate = confident & (entropy <= c['max_normalized_entropy']) & (divergence <= c['max_teacher_js'])
    if c['require_teacher_correct_both_views']:
        gate &= (p.argmax(-1) == target) & (q.argmax(-1) == target)
    reliability = (gate.float() * (1 - entropy).clamp(0, 1) * torch.exp(-divergence / c['reliability_js_scale'])).detach()
    class_weight = weights[target].pow(c['class_weight_power'])
    weighted_gate = class_weight * reliability
    ce = .5 * (F.cross_entropy(sw, target, weight=weights) + F.cross_entropy(ss, target, weight=weights))
    # Cross-view KL: teacher weak -> student strong and vice versa.
    kd_per_image = .5 * temperature**2 * (
        F.kl_div(F.log_softmax(ss / temperature, -1), F.softmax(tw / temperature, -1), reduction='none').sum(-1) +
        F.kl_div(F.log_softmax(sw / temperature, -1), F.softmax(ts / temperature, -1), reduction='none').sum(-1))
    denominator = class_weight.sum().clamp_min(eps)
    kd = (weighted_gate * kd_per_image).sum() / denominator
    consistency = (weighted_gate * js_divergence(sw.softmax(-1), ss.softmax(-1), eps)).sum() / denominator
    total = c['lambda_ce'] * ce + c['lambda_kd'] * kd + c['lambda_consistency'] * consistency
    return total, {'ce': ce.detach(), 'kd': kd.detach(), 'consistency': consistency.detach(),
                   'gate_acceptance': gate.float().mean().detach(), 'gate_mass': reliability.mean().detach(),
                   'gate_mask': gate.detach()}
