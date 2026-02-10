import numpy as np
import h5py as h5
import json
import importlib.resources as resources


antennaConfigPath = resources.files('lambda_commissioning.data')
# Antenna mapping dictionary.
mappingFile = "LAMBDA36-antenna-mappings.json"
with antennaConfigPath.joinpath(mappingFile).open("r") as f:
    antennaDict = json.load(f)
    antennaIDs = np.array(list(antennaDict.keys())).astype(int)

def read_hdf5_data_capture(filePath,verbose=False,returnCorrMatrix=False,
                           flagZeros=True):
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
        #
        visXXtensor = dset[:,:,:,0,0,0] + 1j*dset[:,:,:,0,0,1]
        visYYtensor = dset[:,:,:,-1,-1,0] + 1j*dset[:,:,:,-1,-1,1]

        if np.any(blineIDs<0):
            if verbose:
                print("Negative blineIDs present, indicate non-data, removing.")
            visXXtensor = visXXtensor[:,:,blineIDs>0]
            visYYtensor = visYYtensor[:,:,blineIDs>0]
            blineIDs = blineIDs[blineIDs>0]

    # Assuming the same for XX and YY.
    if flagZeros:
        # Make sure there are no zero antennas.
        zeroBoolVec = visXXtensor[0,0,:] != 0
        visXXtensor = visXXtensor[:,:,zeroBoolVec]
        visYYtensor = visYYtensor[:,:,zeroBoolVec]
        blineIDs = blineIDs[zeroBoolVec]

    if returnCorrMatrix:
        # If True convert the visibility tensors into a correlation matrixß
        # form, with Ntime,Nchan,Nant,Nant, being the output shape.
        antIDs1,antIDs2 = split_baseline(blineIDs)
        
        # Checking that there are no invalid values:
        antIDs1BoolVec = (antIDs1 >= antennaIDs[0])*(antIDs1 <= antennaIDs[-1])
        antIDs2BoolVec = (antIDs2 >= antennaIDs[0])*(antIDs2 <= antennaIDs[-1])
        antIDboolVec = antIDs1BoolVec*antIDs2BoolVec
        
        antIDs1 = antIDs1[antIDboolVec]
        antIDs2 = antIDs2[antIDboolVec]

        antPairs = [(int(ant1),int(antIDs2[ind])) \
                    for ind,ant1 in enumerate(antIDs1)]
        # Also need to remove these correlations from the visibility tensor.
        visXXtensor = visXXtensor[:,:,antIDboolVec]
        visYYtensor = visYYtensor[:,:,antIDboolVec]

        visXXcorrMatrix = make_correlation_tensor(visXXtensor,antPairs)
        visYYcorrMatrix = make_correlation_tensor(visYYtensor,antPairs)

        return visXXcorrMatrix,visYYcorrMatrix,antPairs
    else:
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
    antIDvec = np.unique(antPairs)
    Nants = antIDvec.size

    # Creating a dictionary that maps the antenna index to its name.
    antNameDict = {f"{ant}" : ind for ind,ant in enumerate(antIDvec)}

    corrTensor = np.zeros((Ntimes,Nchans,Nants,Nants),dtype=np.complex64)
    for blInd,(ant1,ant2) in enumerate(antPairs):
        # Getting the index that matches the antenna name.
        ind1,ind2 = antNameDict[str(ant1)],antNameDict[str(ant2)]
        corrTensor[:,:,ind1,ind2] = visTensor[:,:,blInd]
        corrTensor[:,:,ind2,ind1] = np.conj(visTensor[:,:,blInd])

    return corrTensor

def calc_phase_deviation(corrTensor):
    """calc_median_phase_deviation _summary_

    Parameters
    ----------
    corrTensor : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    phaseDeviationMatrix = np.std(np.angle(corrTensor,deg=True),axis=0)
    
    return phaseDeviationMatrix

def calc_amplitude_deviation(corrTensor):
    """calc_median_phase_deviation Calculates the std for a given correlation 
    matrix over time for all channels.

    Parameters
    ----------
    corrTensor : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    amplitudeDeviationMatrix = np.std(np.abs(corrTensor),axis=0)
    
    return amplitudeDeviationMatrix

def calc_median_phase_deviation(corrTensor):
    """calc_median_phase_deviation Calculates the std for a given correlation 
    matrix over time for all channels.

    Parameters
    ----------
    corrTensor : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    #phaseDeviationMatrix = np.median(np.std(np.angle(corrTensor,
    #                                                 deg=True),axis=0),axis=0)
    phaseDeviationMatrix = np.min(np.std(np.angle(corrTensor,
                                                     deg=True),axis=0),axis=0)
    
    return phaseDeviationMatrix

def calc_median_amplitude_deviation(corrTensor):
    """calc_median_phase_deviation _summary_

    Parameters
    ----------
    corrTensor : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    #amplitudeDeviationMatrix = np.median(np.std(np.abs(corrTensor),
    #                                            axis=0),axis=0)
    amplitudeDeviationMatrix = np.min(np.std(np.abs(corrTensor),
                                                axis=0),axis=0)
    
    return amplitudeDeviationMatrix