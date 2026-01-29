import typer
from typing_extensions import Annotated
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from lambda_commissioning.plotting import waterfallPlot,fringePlot
from lambda_commissioning.utils import read_hdf5_data_capture,split_baseline,make_correlation_tensor
import os,sys
from scipy.stats import iqr
import toml
import json
import importlib.resources as resources

configFile = "default_config.toml"
configPath = "lambda_commissioning.config"
antennaConfigPath = resources.files('lambda_commissioning.data')

with resources.files(configPath).joinpath(configFile).open("r") as f:
    config = toml.load(f)

directoryDict = config.get("paths", {})

dataPath = directoryDict['dataPath']
outPath = directoryDict['outputPath']


# Antenna mapping dictionary.
mappingFile = "LAMBDA36-antenna-mappings.json"
with antennaConfigPath.joinpath(mappingFile).open("r") as f:
    antennaDict = json.load(f)


diagnosticApp = typer.Typer(pretty_exceptions_enable=False)

@diagnosticApp.command()
def file_header(filename: Annotated[str,typer.Argument(help="Data filename.")] = ""):
    import h5py as h5

    with h5.File(dataPath+filename, 'r') as hdf_file:
        dset = hdf_file['visibilities']
        print("File keys:", list(hdf_file.keys()))
        print("File attrs:", list(hdf_file.attrs.keys()))
        print("Dataset shape:", dset.shape)
        print("Dataset attrs:",list(dset.attrs.keys()))

@diagnosticApp.command()
def stats(filename: Annotated[str,typer.Argument(help="Data filename.")] = "",
          verbose: Annotated[bool,typer.Option("-v","--verbose",
                                               help="Test optional arg.")] = False,
          plot: Annotated[bool,typer.Option("-p","--plot",
                                               help="Plot the deviation matrices.")] = False,
          channel: Annotated[int,typer.Option("-c","--channel",
                                              help="Starting channel of the observation.")] = None):
    from lambda_commissioning.utils import calc_median_amplitude_deviation
    from lambda_commissioning.utils import calc_median_phase_deviation
    from lambda_commissioning.utils import split_baseline,make_correlation_tensor
    
    # Creating the output directory name.
    outputDir = outPath + f"{filename.split('.')[0]}/"
    outputStatsDir = outputDir + "stats/"

    # Checking if the output directory exists. If not create it. Should always
    # be created in the output directory, assumed to be one level up. 
    if not(os.path.exists(outputDir)):
        os.mkdir(outputDir)
        os.mkdir(outputStatsDir)
        if verbose:
            print(f"Making output directories for files {outputDir}")
    else:
        # Directory might exist, but sub directories might not.
        if not(os.path.exists(outputStatsDir)):
            os.mkdir(outputStatsDir)
            print(f"Making output directory {outputStatsDir}")

    if filename == "":
        raise ValueError("Input filename.")
    
    filePath = dataPath + filename
    if not(os.path.exists(filePath)):
        raise FileNotFoundError(f"No file {filePath}.")
    # Reading in the data.
    visXXtensor,visYYtensor,antPairs = read_hdf5_data_capture(filePath,
                                                              verbose=verbose,
                                                              returnCorrMatrix=True)
  
    # Calculating the median phase and amplitude deviation per channel.
    ampDevMatrixXX = calc_median_amplitude_deviation(visXXtensor)
    phaseDevMatrixXX = calc_median_phase_deviation(visXXtensor)
    ampDevMatrixYY = calc_median_amplitude_deviation(visYYtensor)
    phaseDevMatrixYY = calc_median_phase_deviation(visYYtensor)


    fig,axs = plt.subplots(2,2,figsize=(12,10),constrained_layout=True,
                               sharex=True,sharey=True)

    im1 = axs[0,0].imshow(ampDevMatrixXX,norm='log')
    cb1 = fig.colorbar(im1,ax=axs[0,0],label='Amp [arb units]')
    axs[0,0].set_title('Amp Deviation XX')
    axs[0,0].set_ylabel('AntIDs')


    im2 = axs[0,1].imshow(ampDevMatrixYY,norm='log')
    cb2 = fig.colorbar(im2,ax=axs[0,1],label='Amp [arb units]')
    axs[0,1].set_title('Amp Deviation YY')

    im3 = axs[1,0].imshow(phaseDevMatrixXX,norm='log')
    cb3 = fig.colorbar(im3,ax=axs[1,0],label='Phase [rad]')
    axs[1,0].set_title('Phase Deviation XX')
    axs[1,0].set_xlabel('AntIDs')
    axs[1,0].set_ylabel('AntIDs')

    im4 = axs[1,1].imshow(phaseDevMatrixYY,norm='log')
    cb4 = fig.colorbar(im4,ax=axs[1,1],label='Phase [rad]')
    axs[1,1].set_title('Phase Deviation YY')
    axs[1,1].set_xlabel('AntIDs')

    outFileName = "Amplitude_Phase_Deviation.png"
    fig.savefig(outputStatsDir+outFileName,dpi=300,bbox_inches='tight')
    
    # Saving the deviation matrices.
    outFileNameData = "Amplitude_Phase_Deviation.npz"
    np.savez(outputStatsDir+outFileNameData,ampDevMatrixXX=ampDevMatrixXX,
             ampDevMatrixYY=ampDevMatrixYY,phaseDevMatrixXX=phaseDevMatrixXX,
             phaseDevMatrixYY=phaseDevMatrixYY)

    if plot:
        plt.show()


@diagnosticApp.command()
def autos(filename: Annotated[str,typer.Argument(help="Data filename.")] = "",
          verbose: Annotated[bool,typer.Option("-v","--verbose",
                                               help="Test optional arg.")] = False,
          channel: Annotated[int,typer.Option("-c","--channel",
                                              help="Starting channel of the observation.")] = None):
    # TODO: Save the statistics to a csv file.
    # Creating the output directory name.
    outputDir = outPath + f"{filename.split('.')[0]}/"
    outputAutosDir = outputDir + "autocorrelations/"

    # Checking if the output directory exists. If not create it. Should always
    # be created in the output directory, assumed to be one level up. 
    if not(os.path.exists(outputDir)):
        os.mkdir(outputDir)
        os.mkdir(outputAutosDir)
        if verbose:
            print(f"Making output directories for files {outputDir}")
    else:
        # Directory might exist, but sub directories might not.
        if not(os.path.exists(outputAutosDir)):
            os.mkdir(outputAutosDir)
            print(f"Making output directory {outputAutosDir}")

    if filename == "":
        raise ValueError("Input filename not fiven.")
    
    filePath = dataPath + filename
    if not(os.path.exists(filePath)):
        raise FileNotFoundError(f"No file {filePath}.")

    # Reading in the data.
    visXXtensor,visYYtensor,blineIDs = read_hdf5_data_capture(filePath,
                                                                verbose=verbose)
    # Splitting the baselines into antenna IDs for each baseline.
    ants1,ants2 = split_baseline(blineIDs)
    antPairs = np.vstack((ants1,ants2)).T
    antIDlist = np.unique(ants1)
    Na = np.unique(ants1).size # Number of antennas.
    Nt = visXXtensor.shape[0] # Number of time steps.
    Nc = visXXtensor.shape[1] # Number of channels.
    Nb = Na*(Na-1)/2 # Number of baselines not including the autos.

    # Reorganising into correlation tensos (matrix for each time and 
    # channel). Not strictly necessary, but useful for calibrating and 
    # performing matrix operations on the correlation data.
    corrTensorXX = make_correlation_tensor(visXXtensor,antPairs)
    corrTensorYY = make_correlation_tensor(visYYtensor,antPairs)
    del visXXtensor,visYYtensor

    # Getting a list of the good and the zeroed antennas:
    corrAutoVec = corrTensorXX[0,0,antIDlist,antIDlist]
    zeroAntInds = antIDlist[corrAutoVec == 0]
    goodAntInds = antIDlist[corrAutoVec != 0]
    Na = len(goodAntInds)
    Nb = Na*(Na-1)/2 # Number of baselines not including the autos.

    if channel is not None:
        channels = np.arange(channel,channel+Nc,Nc)

    if verbose:
        print(outputDir)
        print(filename)
        print(verbose)
        print(Nt,Nc,Nb,Na)

    
    # Generating the waterfall plots for the auto correlations. We do this 
    # for each antenna and for each polarisation.
    W = 8
    cmap = cmr.dusk
    #
    autoXXstdVec = np.zeros((Na,Nc))
    autoYYstdVec = np.zeros((Na,Nc))
    autoXXmeanVec = np.zeros((Na,Nc))
    autoYYmeanVec = np.zeros((Na,Nc))
    
    
    for i,antInd in enumerate(goodAntInds):
        #
        waterFallXX = np.abs(corrTensorXX[:,:,antInd,antInd])
        waterFallYY = np.abs(corrTensorYY[:,:,antInd,antInd])

        # Saving statistics.
        autoXXmeanVec[i,:] = np.nanmedian(waterFallXX,axis=0)
        autoYYmeanVec[i,:] = np.nanmedian(waterFallYY,axis=0)
        autoXXstdVec[i,:] = iqr(waterFallXX,axis=0)
        autoYYstdVec[i,:] = iqr(waterFallYY,axis=0)

        #
        stdXX = iqr(waterFallXX)/1.35
        avgXX = np.nanmedian(waterFallXX)
        waterFallXX[waterFallXX > 3*stdXX+avgXX] = np.nan
        
        #
        stdYY = iqr(waterFallYY)/1.35
        avgYY = np.nanmedian(waterFallYY)
        waterFallYY[waterFallYY > 3*stdYY+avgYY] = np.nan

        # Check the data is not zero. If all zeros then truth value of stats 
        # will be False. If any stat value is True we can generate plots,
        # else the data is all zero and we pass, printing the erronous ant ID.
        if np.any(autoXXmeanVec[i,:]) and np.any(autoXXstdVec[i,:]):
            #
            fig,axs = plt.subplots(1,figsize=(2*W,W),sharex=True,
                                constrained_layout=True)
            outFileNameXX = f"auto_correlation_ant{antInd}_polXX.png"
            
            waterfallPlot(waterFallXX,cmap=cmap,title=f'AntID: {antInd}, XX',
                        figaxs=(fig,axs))
            fig.savefig(outputAutosDir+outFileNameXX,dpi=300,bbox_inches='tight')
            plt.close()

            if verbose:
                print(outputAutosDir+outFileNameXX)
        else:
            print(f"XX waterfall is zero for {antInd}")
        
        #
        if np.any(autoYYmeanVec[i,:]) and np.any(autoYYstdVec[i,:]):
            fig,axs = plt.subplots(1,figsize=(2*W,W),sharex=True,
                                constrained_layout=True)
            outFileNameYY = f"auto_correlation_ant{antInd}_polYY.png"
            waterfallPlot(waterFallYY,cmap=cmap,title=f'AntID: {antInd}, YY',
                        figaxs=(fig,axs))
            fig.savefig(outputAutosDir+outFileNameYY,dpi=300,bbox_inches='tight')
            plt.close()
            if verbose:
                print(outputAutosDir+outFileNameYY)
        else:    
            print(f"YY waterfall is zero for {antInd}")
        

@diagnosticApp.command()
def vis(filename: Annotated[str,typer.Argument(help="Data filename.")] = "",
        verbose: Annotated[bool,typer.Option("-v","--verbose",
                                             help="Test optional arg.")] = False,
        channel: Annotated[int,typer.Option("-c","--channel",
                                            help="Starting channel of the observation.")] = None):

    # Creating the output directory name.
    outputDir = outPath + f"{filename.split('.')[0]}/"
    outputAmpDir = outputDir + "vis_amps/"
    outputPhaseDir = outputDir + "vis_phase/"
    outputFringeDir = outputDir + "vis_fringes/"

    # Checking if the output directory exists. If not create it. Should always
    # be created in the output directory, assumed to be one level up. 
    if not(os.path.exists(outputDir)):
        os.mkdir(outputDir)
        os.mkdir(outputAmpDir)
        os.mkdir(outputPhaseDir)
        os.mkdir(outputFringeDir)
        if verbose:
            print(f"Making output directories for files {outputDir}")
    else:
        # Directory might exist, but sub directories might not.
        if not(os.path.exists(outputAmpDir)):
            os.mkdir(outputAmpDir)
            print(f"Making output directory {outputAmpDir}")
        
        if not(os.path.exists(outputPhaseDir)):
            os.mkdir(outputPhaseDir)
            print(f"Making output directory {outputPhaseDir}")
        
        if not(os.path.exists(outputFringeDir)):
            os.mkdir(outputFringeDir)
            print(f"Making output directory {outputFringeDir}")
        

    if filename == "":
        raise ValueError("Input filename not given.")
    
    filePath = dataPath + filename
    if not(os.path.exists(filePath)):
        raise FileNotFoundError(f"No file {filePath}.")
    # Reading in the data.
    corrTensorXX,corrTensorYY,antPairs = read_hdf5_data_capture(filePath,
                                                                verbose=verbose,
                                                                returnCorrMatrix=True)

    antIDlist = np.unique(antPairs)
    Na = antIDlist.size # Number of antennas.
    Nt = corrTensorXX.shape[0] # Number of time steps.
    Nc = corrTensorXX.shape[1] # Number of channels.
    antIndVec = np.arange(Na)
    if channel is not None:
        channels = np.arange(channel,channel+Nc,Nc)

    if verbose:
        print(outputDir)
        print(filename)
        print(verbose)
        print(Nt,Nc,Na)

    ###
    for ant1 in antIndVec:
        ant2 = np.random.choice(np.delete(antIndVec,ant1),size=1)[0]
        antID1 = antIDlist[ant1]
        antID2 = antIDlist[ant2]

        if channel is not None:
            titleXX = f"pol:XX, channel={channels[0]}, antID1={antID1}, " +\
                f"antID2={antID2}, blineID={256*antID1+antID2}"
            titleYY = f"pol:YY, channel={channels[0]}, antID1={antID1}, " +\
                f"antID2={antID2}, blineID={256*antID1+antID2}"
        else:
            titleXX = f"pol:XX, antID1={antID1}, antID2={antID2}, " +\
                f"blineID={256*antID1+antID2}"
            titleYY = f"pol:YY, antID1={antID1}, antID2={antID2}, " +\
                f"blineID={256*antID1+antID2}"
            
        visWaterfallXX = np.abs(corrTensorXX[:,:,ant1,ant2])
        visWaterfallPhaseXX = corrTensorXX[:,:,ant1,ant2]
        stdXX = iqr(visWaterfallXX)/1.35
        avgXX = np.nanmedian(visWaterfallXX)
        visWaterfallXX[visWaterfallXX > 2*stdXX+avgXX] = np.nan
        visWaterfallPhaseXX[visWaterfallXX > 2*stdXX+avgXX] = np.nan
        
        if np.any(avgXX) and np.any(stdXX):
            W = 8
            cmap = cmr.dusk

            fig,axs = plt.subplots(1,figsize=(2*W,W),sharex=True,
                                    constrained_layout=True)
            waterfallPlot(visWaterfallXX,cmap=cmap,title=titleXX,
                        figaxs=(fig,axs))
            outFileNameXXamp = f"vis_amp_waterfall_ant1_{antID1}_ant2_{antID2}_polXX.png"
            fig.savefig(outputAmpDir+outFileNameXXamp,dpi=300,bbox_inches='tight')
            plt.close()
                
            #
            fig,axs = plt.subplots(1,figsize=(2*W,W),sharex=True,
                                    constrained_layout=True)
            waterfallPlot(visWaterfallPhaseXX,cmap=cmr.wildfire,title=titleXX,
                          figaxs=(fig,axs),phaseCond=True,norm='linear')
            outFileNameXXphase = f"vis_phase_waterfall_ant1_{antID1}_ant2_{antID2}_polXX.png"
            fig.savefig(outputPhaseDir+outFileNameXXphase,dpi=300,bbox_inches='tight')
            plt.close()
        
            #
            fig,axs = plt.subplots(1,figsize=(10,5))
            fringePlot(corrTensorXX[:,0,ant1,ant2],figaxs=(fig,axs))
            axs.set_title(titleXX)
            outFileNameXXfringe = f"vis_fringe_ant1_{antID1}_ant2_{antID2}_polXX.png"
            fig.savefig(outputFringeDir+outFileNameXXfringe,dpi=300,bbox_inches='tight')
            plt.close()

            if verbose:
                print(outputAmpDir+outFileNameXXamp)
                print(outputPhaseDir+outFileNameXXphase)
                print(outputFringeDir+outFileNameXXfringe)
        else:
            print(avgXX,stdXX)
            print(visWaterfallXX[:,-1])
            print(f"XX vis is zero for {antID1}")

        #
        visWaterfallYY = np.abs(corrTensorYY[:,:,ant1,ant2])
        visWaterfallPhaseYY = corrTensorYY[:,:,ant1,ant2]
        stdYY = iqr(visWaterfallYY)/1.35
        avgYY = np.nanmedian(visWaterfallYY)
        visWaterfallYY[visWaterfallYY > 2*stdYY+avgYY] = np.nan
        visWaterfallPhaseYY[visWaterfallYY > 2*stdYY+avgYY] = np.nan
        
        #
        if np.any(avgYY) and np.any(stdYY):
            fig,axs = plt.subplots(1,figsize=(2*W,W),sharex=True,
                                    constrained_layout=True)
            waterfallPlot(visWaterfallYY,cmap=cmap,title=titleYY,
                          figaxs=(fig,axs))
            outFileNameYYamp = f"vis_amp_waterfall_ant1_{antID1}_ant2_{antID2}_polYY.png"
            fig.savefig(outputAmpDir+outFileNameYYamp,dpi=300,bbox_inches='tight')
            plt.close()

            #
            fig,axs = plt.subplots(1,figsize=(2*W,W),sharex=True,
                                    constrained_layout=True)
            waterfallPlot(visWaterfallPhaseYY,cmap=cmr.wildfire,title=titleYY,
                          figaxs=(fig,axs),phaseCond=True,norm='linear')
            outFileNameYYphase = f"vis_phase_waterfall_ant1_{antID1}_ant2_{antID2}_polYY.png"
            fig.savefig(outputPhaseDir+outFileNameYYphase,dpi=300,bbox_inches='tight')
            plt.close()
            
            #
            fig,axs = plt.subplots(1,figsize=(10,5))
            fringePlot(corrTensorYY[:,0,ant1,ant2],figaxs=(None,axs))
            axs.set_title(titleYY)
            outFileNameYYfringe = f"vis_fringe_ant1_{antID1}_ant2_{antID2}_polYY.png"
            fig.savefig(outputFringeDir+outFileNameYYfringe,dpi=300,bbox_inches='tight')
            plt.close()

            if verbose:
                print(outputAmpDir+outFileNameYYamp)
                print(outputPhaseDir+outFileNameYYphase)
                print(outputFringeDir+outFileNameYYfringe)
        else:
            print(avgYY,stdYY)
            print(f"YY vis is zero for {antID1}")

        
