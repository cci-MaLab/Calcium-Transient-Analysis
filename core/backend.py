# This is where all the ultility functions are stored

# The first function is to load in the data
import dask.array as darr
import numpy as np
import pandas as pd
import xarray as xr
from os.path import isdir, isfile
from os.path import join as pjoin
from os import listdir
from typing import Callable, List, Optional, Union
import os
import re
import json
import shutil
from dask.diagnostics import ProgressBar

from core.caiman_utils import detrend_df_f, minian_to_caiman

from scipy.signal import welch, savgol_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import find_contours

from matplotlib import cm
import matplotlib.pyplot as plt

import configparser
import time

def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=True
) -> Union[dict, xr.Dataset]:
    """
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


def save_xarray(
    var: xr.DataArray,
    dpath: str,
    compute=True,
) -> xr.DataArray:
    """
    Was having issues with saving xarray to zarr, I think it may have something to do with the
    overwriting data that is not loaded into memory first. Therefore we will first save the data
    as temp then delete the original data and rename the temp data to the original name.
    """
    dpath = os.path.normpath(dpath)
    fp_temp = os.path.join(dpath, var.name + "_temp.zarr")
    fp_orig = os.path.join(dpath, var.name + ".zarr")
    arr = var.to_zarr(fp_temp, compute=compute, mode="w", consolidated=False)
    try:
        shutil.rmtree(fp_orig)
    except FileNotFoundError:
        pass

    # Rename the temp file to the original file
    os.rename(fp_temp, fp_orig)
    
    if compute:
        arr = xr.open_zarr(fp_orig, consolidated=False)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp_orig, var.name), inline_array=True)
    return arr

def delete_missing_xarray(
        dpath: str,
        var_name: str = "M"):
    """
    This is necessary for missing cells as there might be added and removed.
    """
    fp = os.path.join(dpath, var_name + ".zarr")
    try:
        shutil.rmtree(fp)
    except FileNotFoundError:
        pass


def match_information(dpath):# Add by HF
    '''
    Parameters 
    ----------
    str: dpath
        The Session dirctory of mice video
    '''
    pattern_mouseID = "[A-Z]+[0-9]+"
    pattern_day = "D[0-9]+"
    pattern_session = "S\d+"
    pattern1 = r"(/N/project/Cortical_Calcium_Image/Miniscope data/.*?/(?P<mouse_folder_name>.*?))/.*?/.*?/Miniscope_2/(?P<session>S\d+)"
    pattern2 = r"(/N/project/Cortical_Calcium_Image/Miniscope data/.*?/(?P<mouse_folder_name>.*?))/.*?/.*?/Miniscope_2"
    if (re.match(pattern1, dpath)):
        result = re.match(pattern1, dpath)
        mouse_folder_name = result.group("mouse_folder_name")
        mouse_folder_names = mouse_folder_name.split("_")
        if(re.match(pattern_mouseID, mouse_folder_names[0])):
            mouseID = mouse_folder_names[0]
        else:
            raise FileNotFoundError("Wrong mouseID!")
        if(re.match(pattern_day, mouse_folder_names[1])):
            day = mouse_folder_names[1]
        elif(re.match(pattern_day, mouse_folder_names[2])):
            day = mouse_folder_names[2]
        else:
            raise FileNotFoundError("Cannot find mouseID")
        if(re.match(pattern_session, result.group("session"))):
            session = result.group("session")
        else:
            raise FileNotFoundError("Wrong session name!")
        return mouseID, day, session
    elif (re.match(pattern2, dpath)):
        result = re.match(pattern2, dpath)
        mouse_folder_name = result.group("mouse_folder_name")
        mouse_folder_names = mouse_folder_name.split("_")
        if(re.match(pattern_mouseID, mouse_folder_names[0])):
            mouseID = mouse_folder_names[0]
        else:
            raise FileNotFoundError("Wrong mouse name!")
        if(re.match(pattern_day, mouse_folder_names[1])):
            day = mouse_folder_names[1]
        elif(re.match(pattern_day, mouse_folder_names[2])):
            day = mouse_folder_names[2]
        else:
            raise FileNotFoundError("Cannot find mouseID") 
        session = None
        return mouseID, day, session
    else:
        raise FileNotFoundError("Wrong path!")        


def match_path(dpath):# Add by HF
    pattern = r"(?P<video_path>(?P<mouse_path>/N/project/Cortical_Calcium_Image/Miniscope data/.*?/.*?)/.*?/.*?/Miniscope_2)"
    result = re.match(pattern, dpath)
    video_path = result.group("video_path")
    mouse_path = result.group("mouse_path")
    return mouse_path, video_path


        

class Event:
    '''
    Tips:
    1. Use function set_delay_and_duration to set a delay value and a duration value (seconds)
    2. Call function set_switch to set up True or False
    3. Call set_values to pick up the part we want to analysis( Maybe use it on a 'OK' button, or if you want me to change it as automatically, let me know)
    '''
    def __init__(
        self,
        event_type:str,  # ALP, IALP, RNFS
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

    def set_preBinNum(self, preBinNum : int):
        self.preBinNum = preBinNum

    def set_postBinNum(self, postBinNum : int):
        self.postBinNum = postBinNum

    def get_binList(self,event_frame,preBinNum,postBinNum,binSize,value_type):
        binList = []
        total_num = preBinNum + postBinNum
        for i in range(-preBinNum,postBinNum):
            bin, start_frame, end_frame, integrity= self.get_interval_section(event_frame,binSize,i*binSize,0,value_type)
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

        Parameter
        ------------------
        event_frame: int, event time stamp
        duration : float, last time (seconds)
        delay: float, before or after (seconds)
        """
        # duration is in seconds convert to ms
        duration *= 1000
        delay *= 1000
        start = self.data['Time Stamp (ms)'][event_frame]
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
        interval: 100 ms
        '''
         # duration is in seconds convert to ms
        integrity = True
        duration *= 1000
        delay *= 1000
        start = self.data['Time Stamp (ms)'][event_frame]
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
            return self.data[type].sel(frame = frame_list) , event_frame,event_frame + frame_gap, integrity
        else:
            print("No %s data found in minian file" % (type))
            return None
    
        return 

    def set_values(self):
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
        Tips:
        1. load_data and load_events will be automatically excuted.
        2. You may call function set_vector,each time you change the switch of the events.
        3. After you set vectors, call the function compute_clustering, self.linkage_data will be updated. Then you can draw the dendrogram.
        4. Footprint in A. A is a dict. the key is the cell ID.
    '''
    distance_metric_list = ['euclidean','cosine'] # Static variable so parameters can be read before initiating instance
    def __init__(
        self,
        dpath: str
    ):  
        self.events_type = ['ALP','IALP','RNFS','ALP_Timeout']
        self.dpath = dpath  
        self.mouseID : str
        self.day : str
        self.session: str
        self.group: str
        self.minian_path: str
        self.data:dict # Original data, key:'A', 'C', 'S','unit_ids'
        self.events:dict # {"ALP": Event, "IALP" : Event, "RNFS": Event}
        self.A: dict    #key is unit_id,value is A. Just keep same uniform with self.value
        self.value: dict #key is the unit_id,value is the numpy array
        self.outliers_list: List[int] = []
        self.centroids: dict
        self.load_data(dpath=dpath)
        self.no_of_clusters = 4     
        self.distance_metric = 'euclidean'
        self.missed_signals = {}
        self.load_events(self.events_type)
        self.noise_values = {}

        # Create the default image
        self.clustering_result = {"basic": {"image": np.stack((self.data['A'].sum("unit_id").values,)*3, axis=-1)}}

    def add_missed(self, A: np.array):          
        id = max(self.data["M"].coords["missed_id"].values) + 1 if self.data["M"] is not None else 1
        M = xr.DataArray(np.expand_dims(A, axis=0), dims=["missed_id", "height", "width"], coords={"missed_id": [id], "height": self.data['A'].coords["height"].values, "width": self.data['A'].coords["width"].values}, name="M")
        if self.data["M"] is not None:
            M_old = self.data["M"].load()
            M = xr.concat([M_old, M], dim="missed_id")
            

        self.data["M"] = save_xarray(M, self.minian_path, compute=True)
        return id

    def remove_missed(self, ids: List[int]):
        M = self.data["M"].load()
        M = M.drop_sel(missed_id=ids)

        if M.size == 0:
            delete_missing_xarray(self.minian_path, "M")
            self.data["M"] = None
        else:
            self.data["M"] = save_xarray(M, self.minian_path, compute=True)



    def parse_file(self,dpath):# set up configure file
        config = configparser.ConfigParser()
        try:
            config.read(dpath)
        except:
            print("ERROR: ini file is either not in the correct format or empty, did you make sure to save the ini file?")
        if len(config.sections())==1 and config.sections()[0]=='Session_Info':
            return config['Session_Info']['mouseID'],config['Session_Info']['day'],config['Session_Info']['session'],config['Session_Info']['group'], config['Session_Info']['data_path'], config['Session_Info']['behavior_path']
        else:
            print("Error! Section name should be 'Session_Info'!")

    def contains(self, video_type, data_keys):
        for key in data_keys:
            if video_type in key:
                return True, key
        return False, None

    def load_videos(self):
        # We're setting this up as a seperate function as is takes up a lot of space and we only want to load the video info when we need to
        data = open_minian(self.minian_path + "_intermediate")
        video_types = ["Y_fm_chk", "varr", "Y_hw_chk", "behavior_video"]
        video_data = {}
        for video_type in video_types:
            exists, data_type = self.contains(video_type, list(data.keys()))
            if exists:
                video_data[video_type] = data[data_type]
            else:
                print("No %s data found in minian intermediate folder" % (video_type))
        
        self.video_data = video_data       

    def load_data(self,dpath):
        mouseID, day, session, group,minian_path,behavior_path = self.parse_file(dpath)
        self.mouseID = mouseID
        self.day = day
        self.session = session
        self.group = group
        self.minian_path = minian_path
        behavior_data = pd.read_csv(behavior_path,sep=',')
        data_types = ['RNFS', 'ALP', 'IALP', 'ALP_Timeout','Time Stamp (ms)']
        self.data = {}
        for dt in data_types:            
            if dt in behavior_data:
                self.data[dt] = behavior_data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None

        data = open_minian(minian_path)
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

        self.data['unit_ids'] = self.data['C'].coords['unit_id'].values
        self.dpath = dpath

        #zscore 
        # zscore_data = xr.apply_ufunc(
        #     zscore,
        #     self.data['C'].chunk(dict(frame=-1, unit_id="auto")),
        #     input_core_dims=[["frame"]],
        #     output_core_dims=[["frame"]],
        #     dask="parallelized",
        #     output_dtypes=[self.data['C'].dtype],
        # )
        # self.data['C'] = zscore_data
        # self.data['C'] = zscore(self.data['C'], axis = 0)    
    
        self.data['filtered_C'] = self.get_filtered_C

        cells = self.data['unit_ids']

        cent = self.centroid(self.data['A'])

        
        self.A = {}
        self.centroids = {}
        for i in cells:
            self.A[i] = self.data['A'].sel(unit_id = i)
            self.centroids[i] = tuple(cent.loc[cent['unit_id'] == i].values[0][1:])

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
        """
        All positive values are converted to 1.
        """
        a = a.copy()
        a[a > 0] = 1
        return a

    def get_pdf_format(self, unit_ids, cluster, path):
        contours = []
        for id in unit_ids:
            cell = self.A[id].values
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
        the ALP occurs
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
    
    def get_transient_frames(self):
        '''
        Get the inter-event interval. The approach is as follows: the diff of the E array will give us the rising edges.
        For E this means that the start of each transient will have a value of 1. We can extrapolate the inter-event
        by taking their corresponding frame numbers and performing another diff on them.
        '''
        # This will contain 1s for the start of each transient
        rising_edges = self.data["E"].diff(dim="frame").compute()
        # Each cell will have a 1 corresponding to the start of each transient and a
        # a nan value for other frames.
        transient_frames = rising_edges.where(rising_edges==1,drop=True)
        # At this stage it is the most we are able to prune the data. The rest of the 
        # pruning will be done on a per cell basis.
        return transient_frames.compute()
    
    def get_mean_iei_per_cell(self, transient_frames, cell_id, total_transients, frame_rate=None):
        '''
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
        


    def set_vector(self):
        '''
        event :  str, list
            event can be ALP/IALP/RNFS
        '''
        values = {}
        for uid in self.data['unit_ids']:
            values[uid] = np.array([])

        if 'ALP' in self.events.keys():
            for key in self.events['ALP'].values:
                values[key] = np.r_['-1', values[key], self.events['ALP'].values[key]]            
        if 'IALP' in self.events.keys():
            for key in self.events['IALP'].values:
                values[key] = np.r_['-1', values[key], self.events['IALP'].values[key]]
        if 'RNFS' in self.events.keys():
            for key in self.events['RNFS'].values:
                values[key] = np.r_['-1', values[key], self.events['RNFS'].values[key]]
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
        self.cellClustering = CellClustering(self.values,self.outliers_list,self.A,distance_metric = self.distance_metric)
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

    def update_and_save_E(self, unit_id: int, spikes: Union[list, np.ndarray], update_type: str = "Accept Incoming Only"):
        """
        Update the E array with the final peaks and save it to the minian file.
        """
        # First convert final peaks into a numpy array
        E = self.data['E']
        dtype = E.dtype
        if isinstance(spikes, list):
            new_e = np.zeros(E.shape[1], dtype=dtype)
            for spike in spikes:
                new_e[spike[0]:spike[1]] = 1
        else:
            new_e = spikes
        E.load() # Load into memory
        if update_type == "Accept Overlapping Only":
            new_e *= E.sel(unit_id=unit_id).values
        elif update_type == "Accept All":
            new_e += E.sel(unit_id=unit_id).values
            new_e[new_e > 0] = 1
                
        E.loc[dict(unit_id=unit_id)] = new_e
        # Now save the E array to disk
        save_xarray(E, self.minian_path)

    def clear_E(self, unit_id):
        E = self.data['E']
        E.load()
        E.loc[dict(unit_id=unit_id)] = 0
        save_xarray(E, self.minian_path)

    def backup_E(self):
        """
        Backup the E array to disk
        """
        E = self.data['E']
        E.load()
        # Save to backup folder but first check if it exists
        if not os.path.exists(os.path.join(self.minian_path, "backup")):
            os.makedirs(os.path.join(self.minian_path, "backup"))
        t = time.localtime()
        current_time = time.strftime("%m_%d_%H_%M_%S", t)

        
        save_xarray(E, os.path.join(self.minian_path, "backup", "E_" + current_time))

    def remove_from_E(self, clear_selected_events_local: {}):
        E = self.data['E']
        E.load()
        for unit_id, x_values in clear_selected_events_local.items():
            events = E.sel(unit_id=unit_id).values
            events[x_values] = 0          
            E.loc[dict(unit_id=unit_id)] = events
        save_xarray(E, self.minian_path)

    def add_to_E(self, add_selected_events_local: {}):
        E = self.data['E']
        E.load()
        for unit_id, x_values in add_selected_events_local.items():
            events = E.sel(unit_id=unit_id)
            events[x_values] = 1          
            E.loc[dict(unit_id=unit_id)] = events
        save_xarray(E, self.minian_path)
    
    def save_E(self):
        """
        Save the E array to disk
        """
        save_xarray(self.data['E'], self.minian_path)

    def reject_cells(self, cells: List[int]):
        """
        Set the good_cells array to 0 for the cells in the list.
        """
        E = self.data['E']
        E.load()
        E['good_cells'].loc[dict(unit_id=cells)] = 0
        save_xarray(self.data['E'], self.minian_path)

    def approve_cells(self, cells: List[int]):
        E = self.data['E']
        E.load()
        E['good_cells'].loc[dict(unit_id=cells)] = 1
        save_xarray(self.data['E'], self.minian_path)

    def update_verified(self, cells: List[int]):
        E = self.data['E']
        E.load()
        for cell in cells:
            E['verified'].loc[dict(unit_id=cell)] = (E['verified'].loc[dict(unit_id=cell)].values.item() + 1) % 2
        save_xarray(self.data['E'], self.minian_path)
    
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
            self.data['E'] = save_xarray(E, self.minian_path)
            

        # For backwards compatibility check if the verified values exist and if not create them
        elif "verified" not in self.data['E'].coords:
            self.data['E'] = self.data['E'].assign_coords(verified=("unit_id", np.zeros(len(self.data['unit_ids']))))
            save_xarray(self.data['E'], self.minian_path)

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
            self.data['DFF'] = save_xarray(DFF, self.minian_path)


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


        

class CellClustering:
    """
    Cell clustering class. This class is used to cluster cells based on their
    temporal activity, using FFT and agglomerative clustering.
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
        val = self.signals[unit]
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list_pre[unit] = psd
    
    def visualize_dendrogram(self, color_threshold=None, ax=None):
        self.dendro = dendrogram(self.linkage_data,labels=list(self.signals.keys()), color_threshold=color_threshold, ax=ax)
        return self.dendro

    def visualize_clusters(self, t):
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
        viridis = cm.get_cmap('viridis', len(np.unique(self.dendro["leaves_color_list"])))
        
        color_mapping= {}
        for i, leaf in enumerate(self.dendro['leaves']):
            color_mapping[leaf] = int(self.dendro['leaves_color_list'][i][1]) - 1 # Convert to int
        
        image_shape = self.A[list(self.A.keys())[0]].values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx in self.dendro['leaves']:
            final_image += np.stack((self.A[list(self.A.keys())[idx]].values,)*3, axis=-1) * viridis(color_mapping[idx])[:3]
        
        return plt.imshow(final_image)