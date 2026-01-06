import numpy as np
import h5py as h5

def read_hdf5_data_capture(filePath,verbose=False):
    """
    Reads a dataset from an HDF5 file and returns it as a numpy array.

    Parameters
    ----------
    filePath : str
        Path to the HDF5 file.

    Returns
    -------
    data : numpy array
        The dataset read from the HDF5 file.
    """
    with h5.File(filePath, 'r') as hdf_file:
        dset = hdf_file['visibilities']
        if verbose:
            print("Dataset keys:", list(hdf_file.keys()))
            print("Dataset shape:", dset.shape)
            print("Dataset attributes:", list(dset.attrs.keys()))
            print("File attributes:", list(hdf_file.attrs.keys()))
        blineIDs = hdf_file["baseline_ids"][:]

        if np.any(blineIDs<0):
            if verbose:
                print("Negative blineIDs present, indicate non-data, removing.")
            blineIDs = blineIDs[blineIDs>0]
        #
        visXXtensor = dset[:,:,:,0,0,0] + 1j*dset[:,:,:,0,0,1]
        visYYtensor = dset[:,:,:,-1,-1,0] + 1j*dset[:,:,:,-1,-1,1]

    return visXXtensor,visYYtensor,blineIDs

def split_baseline(baselineIDs):
    """
    Function for determining the antenna IDs from the baseline ID. Baseline
    ID is determined by ant1*256 + ant2.

    Parameters
    ----------
    baselineIDs : ndarray
        Numpy array containing the baseline IDs (ant1*256+ant2).

    Returns
    -------
    ant1 : ndarray
        Numpy array containing the the antenna1 ID.
    ant2 : ndarray
        Numpy array containing the the antenna2 ID.
    """    
    if np.max(baselineIDs) >= 65536:
        ant1 = ((baselineIDs - 65536) // 2048).astype(int)
        ant2 = ((baselineIDs - 65536) % 2048).astype(int)
    else:
        ant1 = (baselineIDs // 256).astype(int)
        ant2 = (baselineIDs % 256).astype(int)

    return ant1,ant2

def make_correlation_tensor(visTensor,antPairs):
    """
    Converts a visibility tensor into a full correlation tensor.

    Parameters
    ----------
    visTensor : numpy array
        Visibility tensor of shape (Ntimes, Nchans, Nblines).
    antPairs : list of tuples
        List of antenna index pairs corresponding to each baseline.

    Returns
    -------
    corrTensor : numpy array
        Correlation tensor of shape (Ntimes, Nchans, Nants, Nants).
    """
    Ntimes = visTensor.shape[0]
    Nchans = visTensor.shape[1]
    Nants = max(max(pair) for pair in antPairs) + 1
    corrTensor = np.zeros((Ntimes,Nchans,Nants,Nants),dtype=np.complex64)
    for blInd,(ant1,ant2) in enumerate(antPairs):
        corrTensor[:,:,ant1,ant2] = visTensor[:,:,blInd]
        corrTensor[:,:,ant2,ant1] = np.conj(visTensor[:,:,blInd])

    return corrTensor
    