import numpy as np

def exponential_binning(data, base=2):
    """
    Bins data into exponentially increasing bins with the specified base.

    Parameters:
    data (array-like): 1-dimensional array of data to be binned.
    base (float): The base of the exponential bins. Default is 2.

    Returns:
    bins (array): The edges of the bins used for binning the data.
    probs (array): The normalized probabilities of data points in each bin.
    """
    # Sort the data in increasing order
    data_sorted = np.sort(data)

    # Determine the number of bins needed to cover the range of the data
    num_bins = int(np.ceil(np.log2(data_sorted[-1] / data_sorted[0]) / np.log2(base)))

    # Calculate the edges of the bins using exponential spacing
    bins = data_sorted[0] * base ** np.arange(num_bins + 1)

    # Bin the data using the exponential bins and calculate the probabilities
    counts, _ = np.histogram(data, bins)
    probs = counts / np.sum(counts)
    x = (bins[:-1] * (bins[1:]-1)) ** 0.5

    return x, probs,bins