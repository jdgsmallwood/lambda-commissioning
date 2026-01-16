import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
import matplotlib

def waterfallPlot(data,times=None,channels=None,figaxs=None,cmap=cmr.dusk,
                  phaseCond=False,norm='log',title=None,vmin=None,vmax=None,
                  **kwargs):
    """
    Waterfall plot function for visibility data and for autocorrelation data.

    Parameters
    ----------
    data : numpy array
        Data to plot in the waterfall.
    times : numpy array, optional
        Time values for the x-axis.
    channels : numpy array, optional
        Channel values for the y-axis.
    figaxs : tuple, optional
        Figure and axes objects to use for plotting.
    cmap : str or colormap, optional
        Colormap to use for the plot.
    phaseCond : bool, optional, default=False
        If True, plot the phase of the data instead of the amplitude.
    norm : str, optional, default='log'
        Normalisation for the color scale. Options are 'log' and 'linear'. 
    """
    
    if figaxs is None:
        fig,axs = plt.subplots(1,figsize=(10,5),constrained_layout=True)
    else:
        fig,axs = figaxs
    
    if channels is not None:
        ymin = channels[0]
        ymax = channels[-1]
    else:
        ymin = 0
        ymax = data.shape[1]
    
    if times is not None:
        xmin = times[0]
        xmax = times[-1]
    else:
        xmin = 0
        xmax = data.shape[0]
    
    # Setting vmin and vmax
    if vmax is None:
        if phaseCond:
            vmax = np.pi
        else:
            vmax = np.nanmax(np.abs(data))
    if vmin is None:
        if phaseCond:
            vmin = -np.pi
        else:
            vmin = np.nanmin(np.abs(data))

    if phaseCond:
        if norm == 'log':
            norm = 'linear'

    # Determining the normalisation.
    if norm == 'linear':
        from matplotlib.colors import Normalize
        if vmax == vmin:
            vmin=0
        norm = Normalize(vmin=vmin,vmax=vmax)
    elif norm == 'log':
        from matplotlib.colors import LogNorm
        if vmax == vmin:
            vmin=1e-1
        norm = LogNorm(vmin=vmin,vmax=vmax)
    
    if np.any(vmin) and np.any(vmax):
        extend='both'
    else:
        extend=None

    extent=[xmin,xmax,ymin,ymax]
    # Setting the bad colours if any.
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad('lightgray',1.)

    if phaseCond:
        clabel = "Phase [radians]"
        im = axs.imshow(np.angle(data).T,norm=norm,cmap=cmap,
                        interpolation='None',aspect='auto',extent=extent
                        ,**kwargs)
    else:
        im = axs.imshow(np.abs(data).T,norm=norm,cmap=cmap,
                        interpolation='None',aspect='auto',extent=extent,
                        **kwargs)
        clabel = "Amplitude [arb. units]"

    cb = fig.colorbar(im,ax=axs,aspect=40,extend=extend)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(clabel,fontsize=14)
    if title is not None:
        axs.set_title(title,fontsize=14)
    axs.tick_params(axis='x',labelsize=12)
    axs.tick_params(axis='y',labelsize=12)
    axs.set_ylabel('Channels',fontsize=14)
    axs.set_xlabel('time [samples]',fontsize=14)


def fringePlot(data,times=None,figaxs=None,amplitudeOnly=False,color='k',**kwargs):
    """
    Plots the fringes of a given visibility data array as a function of time. 
    
    Parameters
    ----------
    data : numpy array
        Visibility data to plot.
    times : numpy array, optional
        Time values for the x-axis.
    figaxs : tuple, optional
        Figure and axes objects to use for plotting.
    amplitudeOnly : bool, optional
        If True, only plot the amplitude.
    color : str, optional
        Color of the plot.
    **kwargs
        Additional keyword arguments passed to matplotlib's plot function.
    """

    if figaxs is None:
        _,axs = plt.subplots(1,figsize=(10,5))
    else:
        _,axs = figaxs

    # Getting the antenna indices.
    #
    if times is not None:
        axs.plot(times,np.abs(data),color=color,zorder=1e3,label='Amplitude',
                 linewidth=3,alpha=0.5,**kwargs)
        if not(amplitudeOnly):
            axs.plot(times,data.real,label='Real',color='tab:red')
            axs.plot(times,data.imag,label='Imaginary',color='tab:blue')
    else:
        axs.plot(np.abs(data),color=color,zorder=1e3,label='Amplitude',
                 linewidth=3,alpha=0.5,**kwargs)
        if not(amplitudeOnly):
            axs.plot(data.real,label='Real',color='tab:red')
            axs.plot(data.imag,label='Imaginary',color='tab:blue')

    axs.set_xlabel('time [s]',fontsize=18)
    axs.set_ylabel('Amplitude [arbitrary units]',fontsize=18)

    [x.set_linewidth(2.) for x in axs.spines.values()]
    axs.grid()

    #axs.set_yscale('log')
    axs.tick_params(axis='x',labelsize=16)
    axs.tick_params(axis='y',labelsize=16)
    axs.legend(fontsize=12)
    