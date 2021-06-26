import torch
import heapq
import numpy as np

# Given a transformer model forward function and an array of tokens, return the "engram" based on the hidden states of the transformer
def build_engram(forward, tokens, shift=10000, factor=20000, rampdown=lambda x:x / 2):
    """Given a transformer model forward function and an array of tokens, return the "engram" based on the hidden states of the transformer

    Parameters
    ----------
    forward : function
        A HuggingFace-style model forward function
    tokens : Tensor
        The tokenized text to encode
    shift : float, optional
        How much to add to the values for normalization
    factor : float, optional
        Divisor for normalization
    rampdown : function, optional
        A function to ramp down the power of tokens while iterating over the hidden states

    Returns
    -------
    engram : array
        The final encoded engram
    """

    # get hidden states
    h = list(forward(input_ids=tokens[:, -512:].long().cuda(), output_hidden_states=True).hidden_states[1:])

    # todo: use rampdown
    f = 0
    fa = 1.0/float(len(h))

    # combine hidden states (token axis)
    # we use double() here to reduce accuracy loss from overflowing. it's safe to go back to float() after the math is done. There is probably a more efficient way to do this
    for layer in range(len(h)):
        f = f + fa
        h[layer] = torch.mean(h[layer].detach().double(), dim=(1, )) * f

    h = torch.sum(torch.stack(h, axis=1)[0], dim=(0, ))

    # note: static values are used here to make sorting more consistent. Previously I normalized per-engram but that reduced the overal accuracy of the sorting
    return ((h + shift) / factor).float().to("cpu").numpy()

# Given a "now" engram, and an array of past engrams, return top_k closest matching engrams
def sort_engrams(now, past, factor=1000.0, epsilon=1e-6, top_k=250, depth=1, do_distance=True):
    """Given a "now" engram, and an array of past engrams, return top_k closest matching engrams

    Parameters
    ----------
    now : dict
        The engram to compare against
    past : list
        The list of past engrams to sort
    factor : float, optional
        A function to ramp down the power of tokens while iterating over the hidden states
    epsilon : float, optional
        A small value to add to engrams during sorting
    top_k : int, optional
        The number of closest matching engrams to return
    depth : int, optional
        How many previous or future memories to check against during sorting
    do_distance: bool, optional
        Should we perform distance calculations? Disabling this is useful if you are doing multiple passes with different depths

    Returns
    -------
    sorted : list
        The final sorted list of top_k closest matching engrams
    """
    now = now["engram"].astype(np.float32)

    # calculate distance between all past engrams and the current engram
    if do_distance:
        for e in range(len(past)):
            past[e]["distance"] = np.sum(np.sqrt((np.abs(past[e]["engram"].astype(np.float32) - now) / factor) + epsilon))

    # return the distance value of a given engram, recursively if depth>1
    def keyer(m):
        if depth == 1:
            return m["distance"]
        else:
            total = 0
            nodeup = m
            nodedown = m

            # calculate distance across n previous and future engrams
            for e in range(depth-1):
                nodeup = nodeup["previous"]
                nodedown = nodedown["next"]
                if nodeup is None or nodeup < 0 or nodedown is None or nodedown:
                    total = total + 100000 # some high penalty (unlinked) TODO: better solution
                    break
                
                # scaling factor for distance to root engram
                f = (2.0 * (e + 1.0))

                if nodeup < 0 or nodedown < 0:
                    total = total + 100000 # some high penalty (unlinked) TODO: better solution
                else:
                    nodeup = past[nodeup]
                    nodedown = past[nodedown]
                    total = total + (nodeup["distance"] / f) + (nodedown["distance"] / f)
            return m["distance"] + total

    # pick top_k smallest values (faster than full sort)
    return heapq.nsmallest(top_k, past, key=keyer)
