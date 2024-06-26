from langchain.chat_models import ChatOpenAI
import requests
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import Extra
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import pexpect
import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
from PIL import Image  
def virtual_imaging(mode):
    # Set the filepath
    file_path = '/home/muller_group/Documents/pytcho_dm852/py4DSTEM_tutorials-main/data/'
    dataset = py4DSTEM.read(file_path+"ptycho_MoS2_bin2.h5")
    dataset.calibration
    dataset.get_dp_mean()
    # Find the center and probe radius
    # Get the probe position and size
    datacube = dataset
    probe_semiangle, probe_qx0, probe_qy0 = datacube.get_probe_size(
        datacube.tree('dp_mean').data,
    )
    # Capture and show virtual ADF
    # Position the detector
    # set the geometry
    center = probe_qx0, probe_qy0
    r_inner = probe_semiangle * 1
    r_outer = probe_semiangle * 6
    radii = r_inner,r_outer

    # overlay selected detector position over mean dp
    dataset.position_detector(
        mode = 'annular',
        geometry = (
            center,
            radii
        )
    )

    # compute
    dataset.get_virtual_image(
        mode = 'annulus',
        geometry = (center,radii),
        name = 'annular_dark_field'
    )

    # show
    #py4DSTEM.show(dataset.tree('annular_dark_field'))
    #((dataset_13nm.tree('annular_dark_field'))).data
    # save everthing except the datacube
    file_path = '/home/muller_group/Documents/pytcho_dm852/py4DSTEM_tutorials-main/data/'
    py4DSTEM.save(
        file_path+'llm_output.h5',
        dataset,
        tree = None,  # this indicates saving everything *under* datacube, but not not datacube itself
        mode = 'o'      # if a file of this name already exists, overwrite it
    )
    # save the image
    plt.imsave('/media/muller_group/Extreme Pro/LLMicroscopy/CALMS/output/'+'llm_output.png', ((dataset.tree('annular_dark_field'))).data)
    print('saving output to ./CALMS/output/')
    img = Image.open('/media/muller_group/Extreme Pro/LLMicroscopy/CALMS/output/'+'llm_output.png')
    # img.resize(68,90)
    img.show()

    # Print the estimated center and probe radius
    print('Estimated probe center is', 'qx = %.2f, qy = %.2f' % (probe_qx0, probe_qy0), 'pixels')
    print('Estimated probe radius is', '%.2f' % probe_semiangle, 'pixels')
        # cbed = file.open()
        # py4dstem.virtual_imaging(cbed)
        # xxx.save() 
        # return np.random(5) 
    return f"{'Estimated probe center =', 'qx = %.2f, qy = %.2f' % (probe_qx0, probe_qy0), 'pixels'},{'Estimated probe radius =', '%.2f' % probe_semiangle, 'pixels'}"

def simulate_probe(defocus):
    probe_size = defocus*2+100

    # Assign users' inputs to variables
    voltage = 80 # unit: kV
    df = defocus # unit: nanometer (nm)
    cs = 0
    alpha_max = 30 # unit: mrad

    # Convert unit if needed
    alpha_max = alpha_max / 1000 # mrad to rad
    df = df * 10 # nm to angstrom
    cs = cs * 10000000 # mm to angstrom

    # Define image size and pixel size. Keep them constant
    N = 512
    dx = 0.1 # angstrom

    dk = 1 / (dx * N) # 1/angstrom

    lambda_ =  12.3986 / np.sqrt((2*511.0 + voltage) * voltage) # unit: angstrom

    kx = np.linspace(-N/2, N/2 - 1, N) * dk
    kX, kY = np.meshgrid(kx, kx)
    kR = np.sqrt(kX**2 + kY**2)
    theta = np.arctan2(kY, kX)

    # Calculate aberration function
    chi = np.zeros((N, N), dtype=np.complex_)
    chi = -np.pi * lambda_ * kR**2 * df + np.pi/2 * lambda_**3 * kR**4 * cs

    # Create the disk mask in probe's Fourier magnitude
    mask = np.logical_and(kR <= alpha_max/lambda_, kR >= 0)

    # Create the probe function
    phase = np.exp(-1j * chi)
    probe = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask * phase)))


    data = np.abs(probe) # use probe magnitude

    center = (len(probe)//2, len(probe)//2)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr
    radial_sum = tbin

    x = np.arange(len(radial_profile))
    radius_rms = np.sqrt(np.sum(x**2*radial_profile*x)/np.sum(radial_profile*x))

    HWHM = np.max(np.where((radial_profile / radial_profile.max()) >=0.5))
    FWHM = (2*HWHM+1)*dx

    probe_size = FWHM*2
    return f"{probe_size}"

# func -> langchain line 229 of chat_app
probe_sim_tool = StructuredTool.from_function(simulate_probe,
                                            name="SimulateProbe",
                                            description="Simulate a probe function in electron microscopy with defocus")

# func -> langchain line 229 of chat_app
virtual_imaging_tool = StructuredTool.from_function(virtual_imaging,
                                            name="Virtual_imaging",
                                            description="Estimate probe center and radius, then perform bright field or dark field virtual imaging in electron microscopy")


class ProbeSim(BaseTool, extra=Extra.allow):
    """
    Tool to simulate a probe in electron microscopy
    """
    name = "setdetector"
    description = "tool to simulate probe function in electron microscopy" 

    def __init__(self):
        super().__init__()

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query_params = query.split(' ')

        defocus = query_params[0]

        print(defocus)
        probe_size = simulate_probe(defocus)

        return probe_size

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class VirtualImagingSim(BaseTool, extra=Extra.allow):
    """
    Tool to simulate a probe in electron microscopy
    """
    name = "setmode"
    description = "tool to estimate probe center and radius, then simulate virtual imaging in different imaging modes of electron microscopy，including bright field and dark field." 
    return_direct = True
    def __init__(self):
        super().__init__()

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query_params = query.split(' ')

        imaging_mode = query_params[0]

        print(imaging_mode)
        virtual_image = virtual_imaging(mode)

        return f"{virtual_image}"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

