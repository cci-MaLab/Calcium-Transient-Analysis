# This is where all the ultility functions are stored

# The first function is to load in the data
import dask.array as darr
import numpy as np
import pandas as pd
import xarray as xr
import zarr as zr
from os.path import isdir, isfile
from os.path import join as pjoin
from os import listdir
from typing import Callable, List, Optional, Union, Dict
import os
from pathlib import Path
from uuid import uuid4
import rechunker
import dask as da
from dask.delayed import optimize as default_delay_optimize
import json
import shutil
from dask.diagnostics import ProgressBar

from .caiman_utils import detrend_df_f, minian_to_caiman

from scipy.signal import welch, savgol_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import find_contours

from matplotlib import cm
import matplotlib.pyplot as plt

import configparser
import time
import datetime

def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=True
) -> Union[dict, xr.Dataset]:
    """
    Taken from https://github.com/denisecailab/minian/blob/f64c456ca027200e19cf40a80f0596106918fd09/minian/utilities.py#L278. 
    The current version of minian has outdated dependencies and is not compatible with this project.

    Load an existing minian dataset.

    If `dpath` is a file, then it is assumed that the full dataset is saved as a
    single file, and this function will directly call
    :func:`xarray.open_dataset` on `dpath`. Otherwise if `dpath` is a directory,
    then it is assumed that the dataset is saved as a directory of `zarr`
    arrays, as produced by :func:`save_minian`. This function will then iterate
    through all the directories under input `dpath` and load them as
    `xr.DataArray` with `zarr` backend, so it is important that the user make
    sure every directory under `dpath` can be load this way. The loaded arrays
    will be combined as either a `xr.Dataset` or a `dict`. Optionally a
    user-supplied custom function can be used to post process the resulting
    `xr.Dataset`.

    Parameters
    ----------
    dpath : str
        The path to the minian dataset that should be loaded.
    post_process : Callable, optional
        User-supplied function to post process the dataset. Only used if
        `return_dict` is `False`. Two arguments will be passed to the function:
        the resulting dataset `ds` and the data path `dpath`. In other words the
        function should have signature `f(ds: xr.Dataset, dpath: str) ->
        xr.Dataset`. By default `None`.
    return_dict : bool, optional
        Whether to combine the DataArray as dictionary, where the `.name`
        attribute will be used as key. Otherwise the DataArray will be combined
        using `xr.merge(..., compat="no_conflicts")`, which will implicitly
        align the DataArray over all dimensions, so it is important to make sure
        the coordinates are compatible and will not result in creation of large
        NaN-padded results. Only used if `dpath` is a directory, otherwise a
        `xr.Dataset` is always returned. By default `False`.

    Returns
    -------
    ds : Union[dict, xr.Dataset]
        The resulting dataset. If `return_dict` is `True` it will be a `dict`,
        otherwise a `xr.Dataset`.

    See Also
    -------
    xarray.open_zarr : for how each directory will be loaded as `xr.DataArray`
    xarray.merge : for how the `xr.DataArray` will be merged as `xr.Dataset`
    """
    if isfile(dpath):
        ds = xr.open_dataset(dpath).chunk()
    elif isdir(dpath):
        dslist = []
        for d in listdir(dpath):
            arr_path = pjoin(dpath, d)
            if isdir(arr_path):
                if not arr_path.endswith("backup"):
                    arr = list(xr.open_zarr(arr_path, consolidated=False).values())[0]
                    arr.data = darr.from_zarr(
                        os.path.join(arr_path, arr.name), inline_array=True
                    )
                    dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="override")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds

def save_minian(
    var: xr.DataArray,
    dpath: str,
    meta_dict: Optional[dict] = None,
    overwrite=False,
    chunks: Optional[dict] = None,
    compute=True,
    mem_limit="500MB",
) -> xr.DataArray:
    """
    Taken from https://github.com/denisecailab/minian/blob/f64c456ca027200e19cf40a80f0596106918fd09/minian/utilities.py#L440.
    The current version of minian has outdated dependencies and is not compatible with this project,
    hence the function has been copied here.

    Save a `xr.DataArray` with `zarr` storage backend following minian
    conventions.

    This function will store arbitrary `xr.DataArray` into `dpath` with `zarr`
    backend. A separate folder will be created under `dpath`, with folder name
    `var.name + ".zarr"`. Optionally metadata can be retrieved from directory
    hierarchy and added as coordinates of the `xr.DataArray`. In addition, an
    on-disk rechunking of the result can be performed using
    :func:`rechunker.rechunk` if `chunks` are given.

    Parameters
    ----------
    var : xr.DataArray
        The array to be saved.
    dpath : str
        The path to the minian dataset directory.
    meta_dict : dict, optional
        How metadata should be retrieved from directory hierarchy. The keys
        should be negative integers representing directory level relative to
        `dpath` (so `-1` means the immediate parent directory of `dpath`), and
        values should be the name of dimensions represented by the corresponding
        level of directory. The actual coordinate value of the dimensions will
        be the directory name of corresponding level. By default `None`.
    overwrite : bool, optional
        Whether to overwrite the result on disk. By default `False`.
    chunks : dict, optional
        A dictionary specifying the desired chunk size. The chunk size should be
        specified using :doc:`dask:array-chunks` convention, except the "auto"
        specifiication is not supported. The rechunking operation will be
        carried out with on-disk algorithms using :func:`rechunker.rechunk`. By
        default `None`.
    compute : bool, optional
        Whether to compute `var` and save it immediately. By default `True`.
    mem_limit : str, optional
        The memory limit for the on-disk rechunking algorithm, passed to
        :func:`rechunker.rechunk`. Only used if `chunks` is not `None`. By
        default `"500MB"`.

    Returns
    -------
    var : xr.DataArray
        The array representation of saving result. If `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays. Otherwise all computation leading to the input `var` will
        be preserved in the result.

    Examples
    -------
    The following will save the variable `var` to directory
    `/spatial_memory/alpha/learning1/minian/important_array.zarr`, with the
    additional coordinates: `{"session": "learning1", "animal": "alpha",
    "experiment": "spatial_memory"}`.

    >>> save_minian(
    ...     var.rename("important_array"),
    ...     "/spatial_memory/alpha/learning1/minian",
    ...     {-1: "session", -2: "animal", -3: "experiment"},
    ... ) # doctest: +SKIP
    """
    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    arr = ds.to_zarr(fp, compute=compute, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        with da.config.set(
            array_optimize=darr.optimization.optimize,
            delayed_optimize=default_delay_optimize,
        ):
            zstore = zr.open(fp)
            rechk = rechunker.rechunk(
                zstore[var.name], chunks, mem_limit, dst_path, temp_store=temp_path
            )
            rechk.execute()
        try:
            shutil.rmtree(temp_path)
        except FileNotFoundError:
            pass
        arr_path = os.path.join(fp, var.name)
        for f in os.listdir(arr_path):
            os.remove(os.path.join(arr_path, f))
        for f in os.listdir(dst_path):
            os.rename(os.path.join(dst_path, f), os.path.join(arr_path, f))
        os.rmdir(dst_path)
    if compute:
        arr = xr.open_zarr(fp)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
    return arr


def overwrite_xarray(
    varr: xr.DataArray,
    dpath: str,
    retrieve: bool = False,
) -> xr.DataArray:
    """
    Save an xarray DataArray to a zarr file.

    This function creates a temporary zarr file in the same directory as the
    existing zarr file, and then renames the temporary file to the original. 
    This is due to the fact that certain errors would occur whenever I would
    try to save the zarr file directly to the original file, loading the zarr
    array into memory would also cause the same error. This is a workaround
    to avoid the error.

    Parameters
    ----------

    varr : xr.DataArray
        The xarray DataArray that should be saved.
    dpath : str
        The path to the zarr file that should be saved.
    retrieve : bool, optional
        Whether the saved xarray DataArray should be read from the zarr file.
        By default `False`.

    Returns
    -------
    arr : xr.DataArray
        The saved xarray DataArray. It will identical to the input `varr` but
        it will read from a new zarr file.
    """
    dpath = os.path.normpath(dpath)
    fp_temp = os.path.join(dpath, varr.name + "_temp.zarr")
    fp_orig = os.path.join(dpath, varr.name + ".zarr")
    arr = varr.to_zarr(fp_temp, compute=True, mode="w", consolidated=False)
    try:
        shutil.rmtree(fp_orig)
    except FileNotFoundError:
        pass

    # Rename the temp file to the original file
    _safe_rename(fp_temp, fp_orig)
    if retrieve:
        arr = xr.open_zarr(fp_orig)[varr.name]
        arr.data = darr.from_zarr(os.path.join(fp_orig, varr.name), inline_array=True)
        return arr
    
def _safe_rename(fp_temp, fp_orig, retries=5, delay=0.1):
    # Occasionally on Windows due to permissions errors, the file rename will fail.
    # This function will attempt to rename the file multiple times with a delay to avoid this.
    for _ in range(retries):
        try:
            os.rename(fp_temp, fp_orig)
            return
        except PermissionError:
            time.sleep(delay)
    raise PermissionError(f"Failed to rename {fp_temp} to {fp_orig} after {retries} attempts")


def delete_xarray(
        dpath: str,
        var_name: str = "M") -> None:
    """
    Delete the specified xarray DataArray by removing the zarr file.

    The function serves as a convenience method to deal with "Missing" DataArrays.
    It will be necessary to call this whenever all missing cells have been removed.

    Parameters
    ----------
    dpath : str
        The path to the zarr file that should be deleted.
    var_name : str, optional
        The name of the DataArray that should be deleted. By default "M",
        as we expect this to be the missing data array.
    """
    fp = os.path.join(dpath, var_name + ".zarr")
    try:
        shutil.rmtree(fp)
    except FileNotFoundError:
        pass
      

class Event:
    '''
    An event in this context refers to external behavioral events, such as RNFs, ALPs, ILPs, etc...
    This class also contains various methods to extract relevant information for each Event.

    Parameters
    ----------
    event_type : str
        The type of behavioral event, e.g. "ALP", "ILP", "RNF", etc...
    data : xr.DataArray
        The data array that contains the all CNMF output related data.
    timesteps : List[int]
        A list of timesteps where the event occurs.
    '''
    def __init__(
        self,
        event_type:str,
        data:xr.DataArray,
        timesteps:List[int]
           
    ):  
        self.has_param = False
        self.data = data
        self.event_type = event_type
        self.delay: float
        self.duration: float
        self.switch = False
        self.timesteps = timesteps
        self.values:dict
        self.binSize:int
        self.preBinNum: int
        self.postBinNum: int
        self.binList: list

    def set_binSize(self, binSize: int):
        self.binSize = binSize

    def set_preBinNum(self, preBinNum: int):
        self.preBinNum = preBinNum

    def set_postBinNum(self, postBinNum: int):
        self.postBinNum = postBinNum

    def get_binList(self, event_frame: int, preBinNum: int, postBinNum: int,
                    binSize: int,value_type: int):
        binList = []
        for i in range(-preBinNum,postBinNum):
            bin = self.get_interval_section(event_frame, binSize, i*binSize, 0, value_type)[0]
            binList.append(bin)
        return binList

    def set_delay_and_duration(self, delay:float, duration:float):
        self.delay = delay
        self.duration = duration
        self.has_param = True

    def set_switch(self, switch : bool = True):
        self.switch = switch

    def get_section(self, event_frame: int, duration: float, delay: float = 0.0, type: str = "C") -> xr.Dataset:
        """
        Return the selection of the data that is within the given time frame.
        duration indicates the number of frames.

        Parameters
        ----------
        event_frame: int
            event time stamp
        duration : float
            last time (seconds)
        delay: float
            before or after (seconds)
        """
        # duration is in seconds convert to ms
        duration *= 1000
        delay *= 1000
        max_length = len(self.data['Time Stamp (ms)'])
        if delay > 0:
            frame_gap = 1
            while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] < delay:
                frame_gap += 1
            event_frame += frame_gap
        elif delay < 0:
            frame_gap = -1
            while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] > delay and event_frame + frame_gap > 0:
                frame_gap -= 1
            event_frame += frame_gap
        frame_gap = 1
        while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] < duration and event_frame + frame_gap < max_length-1:
            frame_gap += 1
        if type in self.data:
            return self.data[type].sel(frame=slice(event_frame, event_frame+frame_gap)) , event_frame,event_frame+frame_gap
        else:
            print("No %s data found in minian file" % (type))
            return None

    def get_interval_section(self, event_frame: int, duration: float, delay: float = 0.0, interval:int = 100, type: str = "C") -> xr.Dataset:
        '''
        Return the selection of the data that is within the given time frame.

        Parameters
        ----------
        event_frame: int
            Frame at which the event occurs
        duration : float
            Duration of the event in seconds
        delay: float
            Specifies how much time from the event frame should be included in the selection. 
            If delay is positive, then the selection will start from the event frame + delay. 
            If delay is negative, then the selection will start from the event frame - delay.
        interval: int
            The interval at which the data should be sampled. This is in milliseconds.
        type: str
            Specfies which data type to extract from the minian file. Default is "C".
        '''
        # duration is in seconds convert to ms
        integrity = True
        duration *= 1000
        delay *= 1000
        frame_list = []
        max_length = len(self.data['Time Stamp (ms)'])
        if delay > 0:
            frame_gap = 0
            while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] < delay:
                if (event_frame + frame_gap) < (max_length-1): 
                    frame_gap += 1
                else:
                    integrity = False 
                    break
            event_frame += frame_gap
        elif delay < 0:
            frame_gap = 0
            while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] > delay:
                if(event_frame + frame_gap > 0):
                    frame_gap -= 1
                else:
                    integrity = False
                    break
            event_frame += frame_gap
        frame_gap = 0
        time_flag = self.data['Time Stamp (ms)'][event_frame]
        frame_list.append(event_frame)
        while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] < duration and event_frame + frame_gap < max_length-1:
            if self.data['Time Stamp (ms)'][event_frame + frame_gap]-time_flag > interval:
                time_flag = self.data['Time Stamp (ms)'][event_frame + frame_gap]
                frame_list.append(event_frame + frame_gap)
            frame_gap += 1
        if type in self.data:
            return self.data[type].sel(frame = frame_list) , event_frame, event_frame + frame_gap, integrity
        else:
            print("No %s data found in minian file" % (type))
            return None

    def set_values(self):
        """
        Update the values dictionary with the values of the event data and
        the corresponding windows.
        """
        values={}
        event_list= []
        windows = []
        if self.switch == False:
            self.values=values
            return
        else:
            for i in self.timesteps:
                single_event, start_frame, end_frame = self.get_section(i,self.duration,self.delay)
                event_list.append(single_event)
                windows.append([start_frame, end_frame])
        for i in self.data['unit_ids']:
            values[i] = np.array([])
        for i in event_list:
                for j in i.coords['unit_id'].values:
                    values[j] = np.r_['-1', values[j], np.array(i.sel(unit_id=j).values)]
        self.values = values
        self.windows = windows

class DataInstance:
    '''
    This class is used to store all the data related to a single experiment/recording.
    This includes all CNMF output data, behavioral/timestamp data and video data.

    Parameters
    ----------
    config_path : str
        The path to the configuration file that contains the paths to the minian, behavior and video files.
    
    
    Attributes
    ----------
    events_type : List[str]
        A list of all the event types that are supported by the program.
    mouseID : str
        The mouse ID for the experiment.
    day : str
        The day of the experiment.
    session : str
        The session of the experiment.
    group : str
        The group of the experiment.
    data : dict
        A dictionary that contains all the CNMF output data. The keys are 'A', 'C', 'S', 'E', 'b', 'f', 'DFF', 'YrA', 'M', 'timestamp(ms)'.
    video_data : dict
        A dictionary that contains all the video data. The keys are 'Y_fm_chk', 'varr', 'Y_hw_chk', 'behavior_video'.
    events : dict
        A dictionary that contains all the behavior event data. The keys are the event types and the values are Event objects.
    '''
    distance_metric_list = ['euclidean','cosine'] # Static variable so parameters can be read before initiating instance
    def __init__(
        self,
        config_path: str
    ):  
        self.events_type = ['ALP','ILP','RNF','ALP_Timeout']
        self.config_path = config_path  
        self.mouseID : str
        self.day : str
        self.session: str
        self.group: str

        self.data: dict # Original data, key:'A', 'C', 'S','unit_ids'
        self.video_data: dict # Video data, key: 'Y_fm_chk', 'varr', 'Y_hw_chk', 'behavior_video'
        self.events: dict # {"ALP": Event, "ILP" : Event, "RNF": Event}
        self.outliers_list: List[int] = []

        self.centroids: dict
        self.centroids_to_cell_ids: dict
        self.centroids_max: dict

        self.load_data(config_path=config_path)
        self.no_of_clusters = 4     
        self.distance_metric = 'euclidean'
        self.missed_signals = {}
        self.load_events(self.events_type)
        self.changed_events = False # This is necessary to in the case of recalculating values for sda_widgets
        self.noise_values = {}
        self.cell_ids_to_groups = {} # This differs from self.group as it is used for preselected groups by the user
        # Create the default image
        self.clustering_result = {"basic": {"image": np.stack((self.data['A'].sum("unit_id").values,)*3, axis=-1)}}

    def add_missed(self, A: np.array):
        """
        Adds a missed cell to the data. The missed cell is represented by a footprint mask and
        added to the M data array and saved to the data folder. A unique missed_id is assigned to the missed cell.

        Parameters
        ----------
        A : np.array
            The footprint mask of the missed cell.
        """  
        id = max(self.data["M"].coords["missed_id"].values) + 1 if self.data["M"] is not None else 1
        M = xr.DataArray(np.expand_dims(A, axis=0), dims=["missed_id", "height", "width"], coords={"missed_id": [id], "height": self.data['A'].coords["height"].values, "width": self.data['A'].coords["width"].values}, name="M")
        if self.data["M"] is not None:
            M_old = self.data["M"].load()
            M = xr.concat([M_old, M], dim="missed_id")
            

        self.data["M"] = overwrite_xarray(M, self.cnmf_path, retrieve=True)
        return id

    def remove_missed(self, ids: List[int]):
        """
        Removes the missed cells from the data. The missed cells are identified by their missed_id.

        Parameters
        ----------
        ids : List[int]
            A list of missed_ids that should be removed.
        """
        M = self.data["M"].load()
        M = M.drop_sel(missed_id=ids)

        if M.size == 0:
            delete_xarray(self.cnmf_path, "M")
            self.data["M"] = None
        else:
            self.data["M"] = overwrite_xarray(M, self.cnmf_path, retrieve=True)



    def parse_file(self, config_path):# set up configure file
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
        except:
            print("ERROR: ini file is either not in the correct format or empty, did you make sure to save the ini file?")
        if len(config.sections())==1 and config.sections()[0]=='Session_Info':
            mouseID = config['Session_Info']['mouseID']
            day = config['Session_Info']['day']
            session = config['Session_Info']['session']
            group = config['Session_Info']['group']
            data_path = config['Session_Info']['data_path']
            behavior_path = config['Session_Info']['behavior_path']
            video_path = config['Session_Info']['video_path']
            return mouseID, day, session, group, data_path, behavior_path, video_path
        else:
            print("Error! Section name should be 'Session_Info'!")

    def contains(self, video_type, data_keys):
        for key in data_keys:
            if video_type in key:
                return True, key
        return False, None

    def load_videos(self):
        data = open_minian(self.video_path)
        video_types = ["Y_fm_chk", "varr", "Y_hw_chk", "behavior_video"]
        video_data = {}
        for video_type in video_types:
            exists, data_type = self.contains(video_type, list(data.keys()))
            if exists:
                video_data[video_type] = data[data_type]
            else:
                print("No %s data found in video folder" % (video_type))
                if video_type == "Y_hw_chk" and "Y_fm_chk" in data:
                    print("Creating Y_hw_chk from Y_fm_chk. This may take a while.")
                    Y_hw_chk = save_minian(
                        data["Y_fm_chk"].rename("Y_hw_chk"),
                        self.video_path,
                        overwrite=True,
                        chunks={"frame": -1, "height": 32, "width": 32},
                    )
                    video_data[video_type] = Y_hw_chk
                    print("Done creating Y_hw_chk")
                    
                    
        
        self.video_data = video_data       

    def load_data(self, config_path):
        """
        Load the data from the data path specified in the config file.

        Parameters
        ----------
        config_path : str
            The path to the configuration file that contains the paths to the minian, behavior and video files.
        """
        mouseID, day, session, group, cnmf_path, behavior_path, video_path = self.parse_file(config_path)
        self.mouseID = mouseID
        self.day = day
        self.session = session
        self.group = group
        self.cnmf_path = cnmf_path
        self.video_path = video_path
        behavior_data = pd.read_csv(behavior_path,sep=',')
        data_types = ['RNF', 'ALP', 'ILP', 'ALP_Timeout','Time Stamp (ms)']
        self.data = {}
        for dt in data_types:            
            if dt in behavior_data:
                self.data[dt] = behavior_data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None

        data = open_minian(cnmf_path)
        data_types = ['A', 'C', 'S', 'E', 'b', 'f', 'DFF', 'YrA', 'M','timestamp(ms)']
        timestamp = behavior_data[["Time Stamp (ms)"]]
        timestamp.index.name = "frame"
        da_ts = timestamp["Time Stamp (ms)"].to_xarray()
        data['timestamp(ms)'] = da_ts

        for dt in data_types:            
            if dt in data:
                self.data[dt] = data[dt]
                if "unit_id" in self.data[dt].coords:
                    self.data[dt] = self.data[dt].dropna(dim="unit_id")
                # Safe guard against deprecated E standard and erroneous E data
                if dt == 'E':
                    self.data[dt] = self.data[dt].fillna(0).where(self.data[dt] == 0, 1)
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None

        self.unit_id_consistency()

        self.data['unit_ids'] = np.sort(self.data['C'].coords['unit_id'].values)
        self.config_path = config_path   
    
        self.data['filtered_C'] = self.get_filtered_C

        cells = self.data['unit_ids']

        cent = self.centroid(self.data['A'])
        cent_max = self.centroid_max(self.data['A'])

        
        self.centroids = {}
        self.centroids_max = {}
        for i in cells:
            self.centroids[i] = tuple(cent.loc[cent['unit_id'] == i].values[0][1:])
            self.centroids_max[i] = tuple(cent_max.loc[cent_max['unit_id'] == i].values[0][1:])
        
        self.centroids_to_cell_ids = {v: k for k, v in self.centroids.items()}
        
        
        output_dpath = "/N/project/Cortical_Calcium_Image/analysis"
        if session is None:
            self.output_path = os.path.join(output_dpath, mouseID,day)
        else:
            self.output_path = os.path.join(output_dpath, mouseID,day,session)

        if(os.path.exists(self.output_path) == False):
            os.makedirs(self.output_path)

    def unit_id_consistency(self):
        """
        This function will check if the unit_ids are consistent across all data arrays.
        If not, it will drop the inconsistent unit_ids from all data arrays.
        This will be achieved by taking the intersection of all unit_ids and then filtering
        the data arrays.
        """
        unit_id_list = []
        keys = ['A', 'C', 'S', 'YrA']
        for key in keys:
            unit_id_list.append(set(self.data[key].coords["unit_id"].values))
        
        intersection = set.intersection(*unit_id_list)

        for key in keys:
            self.data[key] = self.data[key].sel(unit_id=list(intersection))
        

    def get_filtered_C(self) -> None:
        """
        This function will filter the C data array by multiplying it with the normalized S data array.
        This has the effect of removing non-event related signals from the C data array.
        """
        normalized_S = xr.apply_ufunc(
            self.normalize_events,
            self.data['S'].chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            output_dtypes=[self.data['E'].dtype],
        )
        filtered_C = self.data['C'] * normalized_S
        return filtered_C
        
    def normalize_events(self, a: np.ndarray) -> np.ndarray:
        # All positive values will be set to 1
        a = a.copy()
        a[a > 0] = 1
        return a

    def get_pdf_format(self, unit_ids, cluster, path):
        contours = []
        for id in unit_ids:
            cell = self.data['A'].sel(unit_id=id).values
            yaoying_param = 6
            thresholded_roi = 1 * cell > (np.mean(cell) + yaoying_param * np.std(cell))
            contours.append(find_contours(thresholded_roi, 0)[0])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        cluster = "all" if cluster == 0 else cluster
        ax.imshow(self.clustering_result[cluster]['image'])
        for cell, unit_id in zip(contours, unit_ids):
            ax.plot(cell[:, 1], cell[:, 0], color='xkcd:azure', alpha=0.5, linewidth=1)
            ax.text(np.mean(cell[:, 1]), np.mean(cell[:, 0]), unit_id, color='xkcd:azure',
                    ha='center', va='center', fontsize="small")
        ax.axis('off')
        fig.savefig(path)

    def get_timestep(self, type: str):
        """
        Return a list that contains contains the a list of the frames where
        the ALP occurs.

        Parameters
        ----------
        type : str
            The type of event to extract the timesteps from.
        """
        return np.flatnonzero(self.data[type])

    def save_justifications(self, justifications):
        filename = "justifications-{self.mouseID}-{self.day}-{self.session}.json"
        with open(os.path.join(self.output_path, filename), "w") as f:
            json.dump(justifications, f)

    def load_justifications(self):
        filename = "justifications-{self.mouseID}-{self.day}-{self.session}.json"
        if os.path.exists(os.path.join(self.output_path, filename)):
            with open(os.path.join(self.output_path, filename), "r") as f:
                return json.load(f)
        else:
            return {}

    
    def load_events(self, keys):
        events = {}
        for key in keys:
            events[key] = Event(key,self.data,self.get_timestep(key))
            events[key].switch = True
        self.events = events

    def get_cell_sizes(self):
        return (self.data["A"] > 0).sum(["height", "width"]).compute()
    
    def get_missed_signal(self, missed_id: int):
        if missed_id in self.missed_signals:
            return self.missed_signals[missed_id]
        mask = self.data["M"].sel(missed_id=missed_id).compute()
        # Extract the dimensions of the mask from x and y axis
        x_range, y_range = self.get_mask_dimensions(mask)
        Y = self.video_data["Y_hw_chk"].sel(height=y_range, width=x_range).compute()
        mask_small = mask.sel(height=y_range, width=x_range)
        averaged_signal = (Y * mask_small).sum(["height", "width"]).compute()
        self.missed_signals[missed_id] = averaged_signal.values

        return self.missed_signals[missed_id]

    def get_mask_dimensions(self, mask):
        # Get mask dimensions but only for positive values
        mask_x = mask.any("height").values
        mask_y = mask.any("width").values
        # Get indices in range
        x_range = np.where(mask_x)[0]
        y_range = np.where(mask_y)[0]

        return x_range, y_range
    
    def get_total_transients(self, unit_id=None):
        # Diff won't capture the first transient, so we add 1 to the sum if
        # the first frame is 1
        if unit_id is None:
            total_transients = (self.data["E"].diff(dim="frame") == 1).sum(dim="frame") + (self.data["E"].isel(frame=0))
            return total_transients.compute()
        else:
            total_transients = (self.data["E"].sel(unit_id = unit_id).diff(dim="frame") == 1).sum(dim="frame") + (self.data["E"].sel(unit_id = unit_id).isel(frame=0))
            return total_transients.compute()
    
    def get_average_peak_dff(self):
        """
        Calculate the average peak dff for each cell. The peak dff is calculated by taking the maximum
        value of the DFF signal of each transient. Then calculate the average of all the peak dffs.

        Returns
        -------
        results : dict
            A dictionary where the keys are the unit_ids and the values are the average peak dffs.
        """
        E = self.data["E"].compute()
        DFF = self.data["DFF"].compute()
        results = {}
        for unit_id in self.data["unit_ids"]:
            peaks = []
            transients = E.sel(unit_id=unit_id).values.nonzero()[0]
            if transients.any():
                # Split up the indices into groups
                transients = np.split(transients, np.where(np.diff(transients) != 1)[0]+1)
                # Now Split the indices into pairs of first and last indices
                transients = [(indices_group[0], indices_group[-1]+1) for indices_group in transients]
            for start, stop in transients:
                peaks.append(DFF.sel(unit_id=unit_id, frame=slice(start, stop)).max().values.item())
            
            if peaks:
                results[unit_id] = np.mean(peaks)
      

        return results


    def get_total_rising_frames(self):
        return (self.data["E"] != 0).sum(dim="frame").compute()
    
    def get_std(self):
        return self.data["DFF"].std(dim="frame").compute()
    
    def get_mad(self, id=None):
        """
        Get the median absolute deviation.

        Parameters
        ----------
        id : int
            The unit_id of the cell for which the MAD should be calculated. If None, then the MAD will be calculated for all cells.
        """
        if id is None:
            median = self.data["DFF"].median(dim="frame").compute()
            mad = abs(self.data["DFF"] - median).median(dim="frame").compute()
        else:
            # It throws an error when trying to extract the median from a single cell therefore convert to numpy first
            dff = self.data["DFF"].sel(unit_id=id).values
            median = np.nanmedian(dff)
            mad = np.nanmedian(abs(dff - median))
        return (1 / 0.6745) * mad
    
    def get_savgol(self, id, params={}):
        """
        Calculate the Savitzky-Golay filter for the DFF signal, this will be used to estimate the noise.

        Parameters
        ----------
        id : int
            The unit_id of the cell for which the Savitzky-Golay filter should be calculated.
        params : dict
            A dictionary that contains the parameters for the Savitzky-Golay filter. The parameters are:

            * win_len : int, optional
                The length of the filter window. Must be an odd integer. Default is 10.
            * poly_order : int, optional
                The polynomial order. Default is 2.
            * deriv : int, optional
                The order of the derivative to compute. Default is 0.
            * delta : float, optional
                The spacing of the samples to which the filter will be applied. Default is 1.0.
            * mode : str, optional
                The mode parameter for the savgol_filter function. Default is "interp".
        """
        window_length = params.get("win_len", 10)
        poly_order = params.get("poly_order", 2)
        deriv = params.get("deriv", 0)
        delta = params.get("delta", 1.0)
        mode = params.get("mode", "interp")


        data = self.data["DFF"].sel(unit_id=id).values
        savgol_data = savgol_filter(data, window_length, poly_order, deriv=deriv, delta=delta, mode=mode)
        return savgol_data
    
    def get_noise(self, savgol_data, id, params={}):
        """
        Noise will be estimated by taking the absolute value difference between the dff data and savgol_smoothed signal.
        The noise will be then estimated with a rolling window approach where the mean, median or maximum value will be taken.

        Parameters
        ----------
        savgol_data : np.array
            The Savitzky-Golay smoothed data.
        id : int
            The unit_id of the cell for which the noise should be calculated.
        """
        noise_type = params.get("type", "Mean")
        win_len = params.get("win_len", 10)
        cap = params.get("cap", 0.01)

        if id in self.noise_values:
            param = noise_type + str(win_len)
            if param in self.noise_values[id]:
                return self.noise_values[id][param]
            else:
                self.noise_values[id] = {}
        
        dff = self.data["DFF"].sel(unit_id=id).values
        noise = abs(dff - savgol_data)
        if noise_type == "Mean":
            noise = np.convolve(noise, np.ones(win_len), 'same') / win_len
        elif noise_type == "Median":
            noise = self.rolling(noise, win_len, "median")
        elif noise_type == "Max":
            noise = self.rolling(noise, win_len, "max")

        noise[noise < cap] = cap
        # May be expensive to compute so save in noise_values
        if id not in self.noise_values:
            self.noise_values[id] = {}
        self.noise_values[id][noise_type + str(win_len)] = noise
        
        return noise
    
    def get_SNR(self, savgol_data, noise):
        """
        We will simply calculate the ratio. However, we will need to make sure that the noise is not 0.
        Any 0 value will be replaced with the lowest non-zero value.

        Parameters
        ----------
        savgol_data : np.array
            The Savitzky-Golay smoothed data.
        noise : np.array
            The noise data.
        """
        # First check if the noise is 0
        if noise.sum() == 0:
            print("ERROR: Noise is 0")
            return noise # Return the noise as the SNR to indicate some sort of error
        
        snr = np.abs(savgol_data) / noise
        # Normalize it so that the SNR max is the savgol_data max
        return snr / snr.max() * savgol_data.max()


    def rolling(self, data, window, rolling_type="median"):
        # Use the pd.Series approach to calculate the rolling window
        s = pd.Series(data)
        if rolling_type == "median":
            return s.rolling(window, center=True, min_periods=1).median().to_numpy()
        elif rolling_type == "max":
            return s.rolling(window, center=True, min_periods=1).max().to_numpy()
    
    def get_transient_frames(self, unit_ids=None):
        '''
        Get the inter-event interval. The approach is as follows: the diff of the E array will give us the rising edges.
        For E this means that the start of each transient will have a value of 1. We can extrapolate the inter-event
        by taking their corresponding frame numbers and performing another diff on them.
        '''
        # This will contain 1s for the start of each transient
        if unit_ids is not None:
            rising_edges = self.data["E"].sel(unit_id=unit_ids).diff(dim="frame").compute()
        else:
            rising_edges = self.data["E"].diff(dim="frame").compute()
        # Each cell will have a 1 corresponding to the start of each transient and a
        # a nan value for other frames.
        transient_frames = rising_edges.where(rising_edges==1,drop=True)
        # At this stage it is the most we are able to prune the data. The rest of the 
        # pruning will be done on a per cell basis.
        return transient_frames.compute()
    
    def get_mean_iei_per_cell(self, transient_frames, cell_id, total_transients, frame_rate=None):
        '''
        Calculate the mean inter-event interval for a single cell. The mean inter-event interval is calculated by taking the
        difference between the start of each transient. The mean is then calculated from the differences.

        Parameters
        ----------
        start_of_transients: xr.DataArray
            The start of each transient for all cells taken as an output of get_mean_iei().
        cell_id: int
            The cell for which we want to calculate the mean inter-event interval
        total_transients: xr.DataArray
            This contains the total number of transients for each cell. The number of transients should 
            correspond to the number of 1s in the frames array. If the length of frames is 1 less than
            the number of transients, then we can assume that the first transient starting at frame 0 was
            missed by the diff operation in get_transient_frames().
        '''

        if cell_id not in transient_frames.coords["unit_id"]:
            return "N/A"
        
        frames = transient_frames.coords["frame"].where(transient_frames.sel(unit_id = cell_id) == 1, drop=True).values
        if total_transients.sel(unit_id = cell_id) == 1:
            return "N/A"
        if len(frames) == total_transients.sel(unit_id = cell_id).item()-1:
            frames = np.insert(frames, 0, 0)

        
        if frame_rate is None:
            return str(round(np.mean(np.diff(frames))))
        else:
            return str(round(np.mean(np.diff(frames))/frame_rate, 3))
        
    def get_transient_frames_iti_dict(self, unit_ids) -> tuple[dict, dict]:
        '''
        Does the same thing as get_transient_frames() but returns two dictionaries.
        The first dictionary contains the unit_ids as keys and the values are the transient frames.
        The second dictionary contains the unit_ids as keys and the values are the inter-event intervals.

        Parameters
        ----------
        unit_ids: List[int]
            The list of unit_ids for which the inter-event interval should be calculated.
        
        Returns
        -------
        frame_start: dict
            A dictionary where the keys are the unit_ids and the values are the start of each transient.
        iti: dict
            A dictionary where the keys are the unit_ids and the values are the inter-event intervals.
        '''
        transient_frames = self.get_transient_frames(unit_ids=unit_ids)

        itis = {}
        frame_start = {}
        for unit_id in unit_ids:
            frame_start[unit_id] = transient_frames.coords["frame"].where(transient_frames.sel(unit_id = unit_id) == 1, drop=True).values
            itis[unit_id] = np.diff(frame_start[unit_id])

        # Calculate differences (IEIs) for each cell
        return frame_start, itis
    



    def set_vector(self):
        values = {}
        for uid in self.data['unit_ids']:
            values[uid] = np.array([])

        if 'ALP' in self.events.keys():
            for key in self.events['ALP'].values:
                values[key] = np.r_['-1', values[key], self.events['ALP'].values[key]]            
        if 'ILP' in self.events.keys():
            for key in self.events['ILP'].values:
                values[key] = np.r_['-1', values[key], self.events['ILP'].values[key]]
        if 'RNF' in self.events.keys():
            for key in self.events['RNF'].values:
                values[key] = np.r_['-1', values[key], self.events['RNF'].values[key]]
        if 'ALP_Timeout' in self.events.keys():
            for key in self.events['ALP_Timeout'].values:
                values[key] = np.r_['-1', values[key], self.events['ALP_Timeout'].values[key]]
        # If no events in this period 
        for uid in self.data['unit_ids']:
            if values[uid].size == 0:
                values[uid] = self.data['C'].sel(unit_id = int(uid)).values
        self.values = values

    def set_distance_metric(self, distance_metirc:str):
        self.distance_metric = distance_metirc

    def set_group(self, group_type: str):
        self.group = group_type

    def set_outliers(self, outliers: List[int]):
        self.outliers_list = outliers

    def set_no_of_clusters(self, number : int):
        self.no_of_clusters = number

    def compute_clustering(self):
        self.cellClustering = CellClustering(self.values, self.outliers_list, self.data["A"], distance_metric = self.distance_metric)
        self.linkage_data = self.cellClustering.linkage_data
        self.clustering_result = self.cellClustering.visualize_clusters(self.no_of_clusters)

    def get_vis_info(self):
        image = self.clustering_result["all"]["image"] if "all" in self.clustering_result else self.clustering_result["basic"]["image"]
        return self.mouseID, self.session, self.day, self.group, image

    def get_dendrogram(self, ax):
        self.cellClustering.visualize_dendrogram(color_threshold =self.linkage_data[(self.no_of_clusters-1),2] ,ax=ax)

    def centroid(self, A: xr.DataArray, verbose=False) -> pd.DataFrame:
        """
        Compute centroids of spatial footprint of each cell.

        Parameters
        ----------
        A : xr.DataArray
            Input spatial footprints.
        verbose : bool, optional
            Whether to print message and progress bar. By default `False`.

        Returns
        -------
        cents_df : pd.DataFrame
            Centroid of spatial footprints for each cell. Has columns "unit_id",
            "height", "width" and any other additional metadata dimension.
        """

        def rel_cent(im):
            im_nan = np.isnan(im)
            if im_nan.all():
                return np.array([np.nan, np.nan])
            if im_nan.any():
                im = np.nan_to_num(im)
            cent = np.array(center_of_mass(im))
            return cent / im.shape

        gu_rel_cent = darr.gufunc(
            rel_cent,
            signature="(h,w)->(d)",
            output_dtypes=float,
            output_sizes=dict(d=2),
            vectorize=True,
        )
        cents = xr.apply_ufunc(
            gu_rel_cent,
            A.chunk(dict(height=-1, width=-1)),
            input_core_dims=[["height", "width"]],
            output_core_dims=[["dim"]],
            dask="allowed",
        ).assign_coords(dim=["height", "width"])
        if verbose:
            print("computing centroids")
            with ProgressBar():
                cents = cents.compute()
        cents_df = (
            cents.rename("cents")
            .to_series()
            .dropna()
            .unstack("dim")
            .rename_axis(None, axis="columns")
            .reset_index()
        )
        h_rg = (A.coords["height"].min().values, A.coords["height"].max().values)
        w_rg = (A.coords["width"].min().values, A.coords["width"].max().values)
        cents_df["height"] = cents_df["height"] * (h_rg[1] - h_rg[0]) + h_rg[0]
        cents_df["width"] = cents_df["width"] * (w_rg[1] - w_rg[0]) + w_rg[0]
        return cents_df
    
    def centroid_max(self, A: xr.DataArray, verbose=False) -> pd.DataFrame:
        """
        Compute the centroid by taking the maximum value in the image. Nearly the same
        as centroid() however it is looks better in the 3D visualizations

        Parameters
        ----------
        A : xr.DataArray
            Input spatial footprints.
        verbose : bool, optional
            Whether to print message and progress bar. By default `False`.

        Returns
        -------
        cents_df : pd.DataFrame
            Centroid Max of spatial footprints for each cell. Has columns "unit_id",
            "height", "width" and any other additional metadata dimension.
        """

        def max_cent(im):
            im_nan = np.isnan(im)
            if im_nan.all():
                return np.array([np.nan, np.nan])
            if im_nan.any():
                im = np.nan_to_num(im)
            max_index_flat = np.argmax(im)
            max_index = np.unravel_index(max_index_flat, im.shape)
            return np.array(max_index)
        
        gu_max_cent = darr.gufunc(
            max_cent,
            signature="(h,w)->(d)",
            output_dtypes=int,
            output_sizes=dict(d=2),
            vectorize=True,
        )

        cents = xr.apply_ufunc(
            gu_max_cent,
            A.chunk(dict(height=-1, width=-1)),
            input_core_dims=[["height", "width"]],
            output_core_dims=[["dim"]],
            dask="allowed",
        ).assign_coords(dim=["height", "width"])

        if verbose:
            print("computing centroids")
            with ProgressBar():
                cents = cents.compute()
        
        cents_df = (
            cents.rename("cents")
            .to_series()
            .dropna()
            .unstack("dim")
            .rename_axis(None, axis="columns")
            .reset_index()
        )

        return cents_df

    def update_and_save_E(self, unit_id: int, spikes: Union[list, np.ndarray], update_type: str = "Accept Incoming Only"):
        """
        Update the E array with the final peaks and save it to the minian file.

        Parameters
        ----------
        unit_id : int
            The unit_id of the cell for which the E array should be updated.
        spikes : Union[list, np.ndarray]
            The final peaks that should be added to the E array.
        update_type : str, optional
            The type of update that should be performed. The options are:
            * Accept Incoming Only : Only accept the incoming spikes and ignore any overlapping spikes.
            * Accept Overlapping Only : Accept all spikes including overlapping spikes.
            * Accept All : Accept all spikes and set the E array to 1 for all the spikes.
        """
        # First convert final peaks into a numpy array
        E = self.data['E']
        dtype = E.dtype
        if isinstance(spikes, list):
            new_e = np.zeros(E.shape[1], dtype=dtype)
            for spike in spikes:
                new_e[spike[0]:spike[1]] = 1
        else:
            new_e = spikes.astype(dtype)
        E.load() # Load into memory
        if update_type == "Accept Overlapping Only":
            new_e *= E.sel(unit_id=unit_id).values
        elif update_type == "Accept All":
            new_e += E.sel(unit_id=unit_id).values
            new_e[new_e > 0] = 1
                
        E.loc[dict(unit_id=unit_id)] = new_e
        # Now save the E array to disk
        overwrite_xarray(E, self.cnmf_path)
        self.changed_events = True
    
    def clear_E(self, unit_id):
        E = self.data['E']
        E.load()
        E.loc[dict(unit_id=unit_id)] = 0
        overwrite_xarray(E, self.cnmf_path)
        self.changed_events = True

    def backup_data(self, name: str):
        """
        Backup a specified data array to the backup folder.

        Parameters
        ----------
        name : str
            The name of the data array to backup.
        """
        data = self.data[name]
        data.load()
        # Save to backup folder but first check if it exists
        if not os.path.exists(os.path.join(self.cnmf_path, "backup")):
            os.makedirs(os.path.join(self.cnmf_path, "backup"))
        t = time.localtime()
        current_time = time.strftime("%m_%d_%H_%M_%S", t)

        
        overwrite_xarray(data, os.path.join(self.cnmf_path, "backup", f"{name}_" + current_time))

    def remove_from_E(self, clear_selected_events_local: Dict[int, List[int]]):
        E = self.data['E']
        E.load()
        for unit_id, x_values in clear_selected_events_local.items():
            events = E.sel(unit_id=unit_id).values
            events[x_values] = 0          
            E.loc[dict(unit_id=unit_id)] = events
        overwrite_xarray(E, self.cnmf_path)
        self.changed_events = True

    def add_to_E(self, add_selected_events_local: Dict[int, List[int]]):
        E = self.data['E']
        E.load()
        for unit_id, x_values in add_selected_events_local.items():
            events = E.sel(unit_id=unit_id)
            events[x_values] = 1          
            E.loc[dict(unit_id=unit_id)] = events
        overwrite_xarray(E, self.cnmf_path)
        self.changed_events = True

    def reject_cells(self, cells: List[int]):
        """
        Set the good_cells array to 0 for the cells in the list.
        """
        E = self.data['E']
        E.load()
        E['good_cells'].loc[dict(unit_id=cells)] = 0
        E['verified'].loc[dict(unit_id=cells)] = 0
        overwrite_xarray(self.data['E'], self.cnmf_path)

    def approve_cells(self, cells: List[int]):
        E = self.data['E']
        E.load()
        E['good_cells'].loc[dict(unit_id=cells)] = 1
        overwrite_xarray(self.data['E'], self.cnmf_path)

    def update_verified(self, cells: List[int], force_verified: bool = False):
        E = self.data['E']
        E.load()
        for cell in cells:
            if force_verified:
                E['verified'].loc[dict(unit_id=cell)] = 1
            else:
                E['verified'].loc[dict(unit_id=cell)] = (E['verified'].loc[dict(unit_id=cell)].values.item() + 1) % 2
        overwrite_xarray(self.data['E'], self.cnmf_path)
    
    def prune_non_verified(self, cells: set):
        # Keep only verified cells in cells
        verified_unit_ids = self.get_verified_cells()

        is_list = False
        if type(cells) == list:
            cells = set(cells)
            is_list = True

        if is_list:
            return list(cells.intersection(verified_unit_ids))
        else:
            return cells.intersection(verified_unit_ids)
    
    def prune_rejected_cells(self, cells):
        """
        Prune the cells that have been rejected from the list of cells.
        """
        E = self.data['E']
        E.load()
        return [cell for cell in cells if E.sel(unit_id=cell)['good_cells'].values.item() == 1]
    
    def get_verified_cells(self):
        all_unit_ids = self.data['E'].unit_id.values
        verified_idxs = self.data['E'].verified.values.astype(int)
        verified_unit_ids = all_unit_ids[verified_idxs==1]

        return verified_unit_ids
    
    def get_good_cells(self):
        all_unit_ids = self.data['E'].unit_id.values
        good_idxs = self.data['E'].good_cells.values.astype(int)
        good_unit_ids = all_unit_ids[good_idxs==1]

        return good_unit_ids


    
    def check_E(self):
        """
        Check if the E xarray exists and if not create it.
        """
        if self.data['E'] is None:
            print("Creating E array")
            E = xr.DataArray(
                np.zeros(self.data['C'].shape),
                dims=["unit_id", "frame"],
                coords=dict(
                    unit_id=self.data['unit_ids'],
                    frame=self.data['C'].coords["frame"],
                ),
                name="E"
            )
            E = E.assign_coords(good_cells=("unit_id", np.ones(len(self.data['unit_ids']))), verified=("unit_id", np.zeros(len(self.data['unit_ids']))))
            E.coords['timestamp(ms)'] = self.data['timestamp(ms)']
            self.data['E'] = overwrite_xarray(E, self.cnmf_path, retrieve=True)
            

        # For backwards compatibility check if the verified values exist and if not create them
        elif "verified" not in self.data['E'].coords:
            self.data['E'] = self.data['E'].assign_coords(verified=("unit_id", np.zeros(len(self.data['unit_ids']))))
            overwrite_xarray(self.data['E'], self.cnmf_path)

    def check_DFF(self):
        """
        Check if the DFF xarray exists and if not create it.
        """
        if self.data['DFF'] is None:
            print("Creating DFF array. Sit tight this could take a while.")
            # Convert the data into caiman format
            A, b, C, f, YrA = minian_to_caiman(self.data['A'], self.data['b'], self.data['C'], self.data['f'], self.data['YrA'])
            dff_array = detrend_df_f(A, b, C, f, YrA, flag_auto=False)
                
            DFF = xr.DataArray(
                dff_array,
                dims=["unit_id", "frame"],
                coords=dict(
                    unit_id=self.data['unit_ids'],
                    frame=self.data['C'].coords["frame"],
                ),
                name="DFF"
            ).chunk(dict(frame=-1, unit_id="auto"))
            self.data['DFF'] = overwrite_xarray(DFF, self.cnmf_path, retrieve=True)


    def check_essential_data(self):
        '''
        Create a list of essential data that is required for the analysis.
        '''
        essential_data = ["A", "C", "S", "b", "f", "YrA"]
        got_data = True

        for data_type in essential_data:
            if self.data[data_type] is None:
                print("Missing essential data: %s" % data_type)
                got_data = False

        return got_data

    def add_cell_id_group(self, cell_ids: List, group_id: str):
        """
        Allocate specific cell ids to a group id

        This function will allocate the cell ids to a group id. This will
        be stored in a dictionary where the key is the cell id and the value
        is a set of group ids. If group_id is an empty string, we will
        allocate a number as the group id.

        Parameters
        ----------
        cell_ids : list
            List of cell ids to allocate to the group
        group_id : str
        """
        if group_id == "":
            # First find the lowest available group id
            current_ids = set(self.cell_ids_to_groups.values())
            group_id = 1
            while group_id in current_ids:
                group_id += 1

        for id in cell_ids:
            if id not in self.cell_ids_to_groups:
                self.cell_ids_to_groups[id] = [group_id]
            else:
                self.cell_ids_to_groups[id] += [group_id]
    
    def remove_cell_id_group(self, cell_id_group: List):
        """
        Remove the cell ids from the group id.
        
        Parameters
        ----------
        cell_id_group : list
            List of cell ids to remove from the group id
        """
        for id in cell_id_group:
            if id in self.cell_ids_to_groups:
                del self.cell_ids_to_groups[id]

    def get_group_ids(self):
        all_group_ids = []
        for group_ids in self.cell_ids_to_groups.values():
            all_group_ids += group_ids
        return np.unique(all_group_ids)

    def get_video_interval(self):
        timestamps = self.data['timestamp(ms)'].values
        # Take first 100 frames and calculate the frame rate
        elapsed_time = timestamps[100] - timestamps[0]
        frame_time = int(elapsed_time / 100)
        return frame_time

    def frame_to_time(self, frame):
        timestamp = self.data['timestamp(ms)'].values[frame].item()
        # Convert to 00:00:00.00 format
        seconds = timestamp / 1000
        isec, fsec = divmod(round(seconds*100), 100)
        return "{}.{:02.0f}".format(datetime.timedelta(seconds=isec), fsec)

    def get_cell_ids(self, group_id, verified=False):
        """
        Get the cell ids for the group id.

        Parameters
        ----------
        group_id : str
            The group id to extract the cell ids from.
        verified : bool
            If True, only extract the verified cells.
        """
        if group_id == "All Cells":
            unit_ids = self.data['E'].unit_id.values
        elif group_id == "Verified Cells":
            all_unit_ids = self.data['E'].unit_id.values
            verified_idxs = self.data['E'].verified.values.astype(int)
            unit_ids = all_unit_ids[verified_idxs==1]
        else:
            if "Group" not in group_id:
                raise ValueError("Invalid group id")
            # Extract the group id from the string
            group_id = group_id[6:]
            # Find the corresponding cell ids
            unit_ids = [key for key, value in self.cell_ids_to_groups.items() if group_id in value]

        if verified:
            all_unit_ids = self.data['E'].unit_id.values
            verified_cells = self.data['E'].verified.values
            verified_cells = all_unit_ids[verified_cells==1]
            intersection = np.intersect1d(unit_ids, verified_cells)
            unit_ids = list(intersection)
        
        unit_ids.sort()
        return unit_ids
    
    def merge_cells(self, cell_ids: List[List[int]]):
        """
        Merge the cells in the list of cell ids. By averaging both their spatial footprints and temporal activities.
        The previous C, S, A, YrA, DFF and E arrays will be first backed up before the merge is performed. The E array 
        will drop the cell ids that are not in the list of cell ids to merge and it will change the verified status
        to 0 for the merged cell id.

        Parameters
        ----------
        cell_ids : list
            List of cell ids to merge.
        """
        # For each group of cells within the list, we'll take the lowest cell id as the main cell id
        # and merge the rest of the cells into it.
        cell_mapping = {}
        for group in cell_ids:
            min_id = min(group)
            for id in group:
                if id != min_id:
                    cell_mapping[id] = min_id

        unit_labels = self.data['unit_ids']
        for i in range(len(unit_labels)):
            if unit_labels[i] in cell_mapping:
                unit_labels[i] = cell_mapping[unit_labels[i]]

        # Backup stuff here
        self.backup_data("A")
        self.backup_data("C")
        self.backup_data("S")
        self.backup_data("YrA")
        self.backup_data("DFF")
        self.backup_data("E")


        # Merge the spatial footprints
        A_merge = (
            self.data["A"].assign_coords(unit_labels=("unit_id", unit_labels))
            .groupby("unit_labels")
            .mean("unit_id")
            .rename(unit_labels="unit_id")
        )

        C_merge = (
            self.data["C"].assign_coords(unit_labels=("unit_id", unit_labels))
            .groupby("unit_labels")
            .mean("unit_id")
            .rename(unit_labels="unit_id")
        )

        S_merge = (
            self.data["S"].assign_coords(unit_labels=("unit_id", unit_labels))
            .groupby("unit_labels")
            .mean("unit_id")
            .rename(unit_labels="unit_id")
        )

        YrA_merge = (
            self.data["YrA"].assign_coords(unit_labels=("unit_id", unit_labels))
            .groupby("unit_labels")
            .mean("unit_id")
            .rename(unit_labels="unit_id")
        )

        DFF_merge = (
            self.data["DFF"].assign_coords(unit_labels=("unit_id", unit_labels))
            .groupby("unit_labels")
            .mean("unit_id")
            .rename(unit_labels="unit_id")
        )

        # For E we will first go through the keys and update verified to 0 for the merged cells
        for key in cell_mapping.keys():
            self.data["E"]["verified"].loc[dict(unit_id=key)] = 0
        
        # Get the all values of cell_mapping into a list and drop them from E
        drop_keys = list(cell_mapping.keys())
        self.data["E"] = self.data["E"].drop_sel(unit_id=drop_keys)

        # Save the new arrays
        self.data["A"] = overwrite_xarray(A_merge, self.cnmf_path, retrieve=True)
        self.data["C"] = overwrite_xarray(C_merge, self.cnmf_path, retrieve=True)
        self.data["S"] = overwrite_xarray(S_merge, self.cnmf_path, retrieve=True)
        self.data["YrA"] = overwrite_xarray(YrA_merge, self.cnmf_path, retrieve=True)
        self.data["E"] = overwrite_xarray(self.data["E"], self.cnmf_path, retrieve=True)
        self.data["DFF"] = overwrite_xarray(DFF_merge, self.cnmf_path, retrieve=True)
        

        

class CellClustering:
    """
    Cell clustering class. This class is used to cluster cells based on their
    temporal activity, using FFT and agglomerative clustering.

    Parameters
    ----------
    section : dict
        A dictionary containing the cell ids as keys and the temporal activity
        as values.
    outliers_list : list
        A list of cell ids that should be excluded from the clustering.
    A : xr.DataArray
        The spatial footprints of the cells.
    fft : bool, optional
        Whether to use FFT to compute the PSD. By default `True`.
    distance_metric : str, optional
        The distance metric to use for the clustering. The options are:
        - euclidean
        - cosine
    
    Attributes
    ----------
    A : xr.DataArray
        The spatial footprints of the cells.
    psd_list_pre : dict
        A dictionary containing the cell ids as keys and the PSD as values.
    psd_list : list
        A list of the PSD values.
    outliers_list : list
        A list of cell ids that should be excluded from the clustering.
    special_unit : list
        A list of cell ids that have no activity.
    distance_metric : str
        The distance metric to use for the clustering.
    signals : dict
        A dictionary containing the cell ids as keys and the temporal activity
        as values.
    linkage_data : np.array
        The linkage data for the clustering.
    dendro : dict
        The dendrogram data.
    cluster_indices : np.array
        The cluster indices.
    """

    def __init__(
        self,
        section: Optional[dict] = None,
        outliers_list: List[int] = [],
        A: Optional[xr.DataArray] = None,
        fft: bool = True,
        distance_metric: str = 'euclidean'
    ):
        self.A = A
        self.psd_list_pre = {}
        self.psd_list = []
        self.outliers_list = outliers_list
        self.special_unit = []
        self.distance_metric = distance_metric
        self.signals = {}
        for values in section.keys():
            if values not in self.outliers_list:
                self.signals[values] = section[values]

        if fft:
            for unit_id in self.signals:
                self.compute_psd(unit_id) # compute psd for each unit
        else:
            self.psd_list = [self.signals[unit_id] for unit_id in self.signals]        
        # Compute agglomerative clustering
        if self.distance_metric == 'euclidean':
            for unit_id in self.psd_list_pre:
                self.psd_list.append(self.psd_list_pre[unit_id])
            self.linkage_data = linkage(self.psd_list, method='average', metric='euclidean')
        elif self.distance_metric == 'cosine':
            for (unit_id,unit_id) in zip(self.signals,self.psd_list_pre):
                if(all(value == 0 for value in self.psd_list_pre[unit_id]) ==True):
                    self.special_unit.append(unit_id)
                else:
                    self.psd_list.append(self.psd_list_pre[unit_id])
            self.linkage_data = linkage(self.psd_list, method='average', metric='cosine')

    def compute_psd(self, unit: int):
        """
        Compute the power spectral density of the signal for a given cell.

        Parameters
        ----------
        unit : int
            The cell id.
        """
        val = self.signals[unit]
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list_pre[unit] = psd
    
    def visualize_dendrogram(self, color_threshold=None, ax=None):
        """
        Apply dendrogram from scipy.cluster.hierarchy and save result to class attribute.

        Parameters
        ----------
        color_threshold : float, optional
            The color threshold for the dendrogram. By default `None`.
        ax : matplotlib.axes.Axes, optional
            The axes to plot the dendrogram. By default `None`.
        
        Returns
        -------
        dendro : dict
            The dendrogram data.
        """
        self.dendro = dendrogram(self.linkage_data,labels=list(self.signals.keys()), color_threshold=color_threshold, ax=ax)
        return self.dendro

    def visualize_clusters(self, t):
        """
        Visualize the clusters by assigning a color to each cluster and looking up the corresponding
        footprint of each cell.

        Parameters
        ----------
        t : int
            The number of clusters to create.
        
        Returns
        -------
        cluster_result : dict
            A dictionary containing the cluster results. The keys are the cluster indices and the values
            are dictionaries containing the cell ids and the image of the cluster.
        """
        self.cluster_indices = fcluster(self.linkage_data, t=t, criterion='maxclust')
        
        viridis = cm.get_cmap('jet', self.cluster_indices.max()+1)        
        image_shape = self.A[list(self.A.keys())[0]].values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))
        
        cluster_result = {}
        cluster_result["all"] = {}
        cluster_result["all"]["ids"] = []
        cluster_result["all"]["image"] = final_image.copy()
        for i in range(t+1):
            cluster_result[i] = {}
            cluster_result[i]["ids"] = []
            cluster_result[i]["image"] = final_image.copy()

        for idx, cluster in enumerate(self.cluster_indices):
            cluster_result[cluster]["ids"].append(list(self.signals.keys())[idx])
            cluster_result["all"]["ids"].append(list(self.signals.keys())[idx])
            cluster_result[cluster]["image"] += np.stack((self.A[list(self.signals.keys())[idx]].values,)*3, axis=-1) * viridis(cluster)[:3]
            final_image += np.stack((self.A[list(self.signals.keys())[idx]].values,)*3, axis=-1) * viridis(cluster)[:3]
        
        cluster_result["all"]["image"] = final_image

        return cluster_result
    
    def visualize_clusters_color(self):
        """
        Slightly different approach to visualize the clusters. This will color the cells based on the cluster
        the dendrogram results.

        Returns
        -------
        matplotlib.image.AxesImage
            The image of the clustered cells.
        """
        viridis = cm.get_cmap('viridis', len(np.unique(self.dendro["leaves_color_list"])))
        
        color_mapping= {}
        for i, leaf in enumerate(self.dendro['leaves']):
            color_mapping[leaf] = int(self.dendro['leaves_color_list'][i][1]) - 1 # Convert to int
        
        image_shape = self.A[list(self.A.keys())[0]].values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx in self.dendro['leaves']:
            final_image += np.stack((self.A[list(self.A.keys())[idx]].values,)*3, axis=-1) * viridis(color_mapping[idx])[:3]
        
        return plt.imshow(final_image)