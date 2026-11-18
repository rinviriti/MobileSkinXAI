"""Pure implementation of the originally prescribed strict-improvement stopping rule."""
def update(best, best_epoch, stale, score, epoch, frozen):
    improved = score > best
    if improved:
        return score, epoch, 0, True
    return best, best_epoch, stale + (not frozen), False

def should_stop(epoch, frozen_epochs, stale, patience):
    return epoch > frozen_epochs and stale >= patience
