# This is where all the ultility functions are stored

# The first function is to load in the data
import dask as da
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
from dask.diagnostics import ProgressBar

from scipy.signal import welch
from scipy.signal import savgol_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.measurements import center_of_mass

from matplotlib import cm
from matplotlib import colors 
import matplotlib.pyplot as plt

def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=False
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
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(
                    os.path.join(arr_path, arr.name), inline_array=True
                )
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds

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
        while self.data['Time Stamp (ms)'][event_frame + frame_gap] - self.data['Time Stamp (ms)'][event_frame] < duration and event_frame + frame_gap < max_length:
            frame_gap += 1
        if type in self.data:
            return self.data[type].sel(frame=slice(event_frame, event_frame+frame_gap)) , event_frame,event_frame+frame_gap
        else:
            print("No %s data found in minian file" % (type))
            return None


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

class SessionFeature:
    '''
        Tips:
        1. load_data and load_events will be automatically excuted.
        2. You may call function set_vector,each time you change the switch of the events.
        3. After you set vectors, call the function compute_clustering, self.linkage_data will be updated. Then you can draw the dendrogram.
        4. Footprint in A. A is a dict. the key is the neuron ID.
    '''
    def __init__(
        self,
        dpath: str,
        events: list
    ):  
        self.dpath = dpath  
        self.mouseID : str
        self.day : str
        self.session: str
        self.group: str
        self.data:dict # Original data, key:'A', 'C', 'S','unit_ids'
        self.events:dict # {"ALP": Event, "IALP" : Event, "RNFS": Event}
        self.A: dict    #key is unit_id,value is A. Just keep same uniform with self.value
        self.value: dict #key is the unit_id,value is the numpy array
        self.outliers_list: List[int] = []
        # self.linkage_data:
        self.load_data(dpath=dpath)
        self.load_events(events)
        self.no_of_clusters = 4

    def load_data(self,dpath):
        mouseID, day, session = match_information(dpath)
        mouse_path, video_path = match_path(dpath)
        self.mouseID = mouseID
        self.day = day
        self.session = session
        if (session is None):
            behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + "behavior_ms.csv"),sep=',')
        else:
            behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + session + "_" + "behavior_ms.csv"),sep=',')
        data_types = ['RNFS', 'ALP', 'IALP', 'ALP_Timeout','Time Stamp (ms)']
        self.data = {}
        for dt in data_types:            
            if dt in behavior_data:
                self.data[dt] = behavior_data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None

        minian_path = os.path.join(dpath, "minian")
        data = open_minian(minian_path)
        data_types = ['A', 'C', 'S']
        for dt in data_types:            
            if dt in data:
                self.data[dt] = data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None
        
        self.data['unit_ids'] = self.data['C'].coords['unit_id'].values
        self.dpath = dpath

        neurons = self.data['unit_ids']
        
        self.A = {}
        for i in neurons:
            self.A[i] = self.data['A'].sel(unit_id = i)

        output_dpath = "/N/project/Cortical_Calcium_Image/analysis"
        if session is None:
            self.output_path = os.path.join(output_dpath, mouseID,day)
        else:
            self.output_path = os.path.join(output_dpath, mouseID,day,session)

        if(os.path.exists(self.output_path) == False):
            os.makedirs(self.output_path)
        

    def get_timestep(self, type: str):
        """
        Return a list that contains contains the a list of the frames where
        the ALP occurs
        """
        return np.flatnonzero(self.data[type])

    
    def load_events(self, keys):
        events = {}
        for key in keys:
            events[key] = Event(key,self.data,self.get_timestep(key))
            events[key].switch = True
        self.events = events

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


    def set_group(self, group_type: str):
        self.group = group_type

    def set_outliers(self, outliers: List[int]):
        self.outliers_list = outliers

    def set_no_of_clusters(self, number : int):
        self.no_of_clusters = number

    def compute_clustering(self):
        self.cellClustering = CellClustering(self.values,self.outliers_list,self.A)
        self.linkage_data = self.cellClustering.linkage_data
        self.clustering_result = self.cellClustering.visualize_clusters(self.no_of_clusters)

    def get_vis_info(self):
        val = int(self.day[1:])
        y = 1 if val < 4 else 2
        x = 1 if self.session == 'S1' else 2

        return self.mouseID, x, y, self.group, self.clustering_result['all']['image']

    def get_dendrogram(self, ax):
        self.cellClustering.visualize_dendrogram(color_threshold =self.linkage_data[(self.no_of_clusters-1),2] ,ax=ax)

    def centroid(A: xr.DataArray, verbose=False) -> pd.DataFrame:
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

        gu_rel_cent = da.gufunc(
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
        fft: bool = True
    ):
        self.A = A
        self.psd_list = []
        self.outliers_list = outliers_list
        self.signals = {}
        for i in section.keys():
            if i not in self.outliers_list:
                self.signals[i] = section[i]

        if fft:
            for unit_id in self.signals:
                self.compute_psd(unit_id) # compute psd for each unit
        else:
            self.psd_list = [self.signals[unit_id] for unit_id in self.signals]        
        # Compute agglomerative clustering
        self.linkage_data = linkage(self.psd_list, method='average', metric='euclidean')

    def compute_psd(self, unit: int):
        val = self.signals[unit]
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list.append(psd)
    
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
            cluster_result[cluster]["ids"].append(list(self.A.keys())[idx])
            cluster_result["all"]["ids"].append(list(self.A.keys())[idx])
            cluster_result[cluster]["image"] += np.stack((self.A[list(self.A.keys())[idx]].values,)*3, axis=-1) * viridis(cluster)[:3]
            final_image += np.stack((self.A[list(self.A.keys())[idx]].values,)*3, axis=-1) * viridis(cluster)[:3]
        
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


# class FeatureExploration:
#     """
#     The purpose of this class is to explore potential features that can be used
#     for the clustering of cells.
#     """
#     def __init__(
#         self,
#         dpath: str,
#     ):        
#         mouseID, day, session = match_information(dpath)
#         mouse_path, video_path = match_path(dpath)
#         print(session)
#         if (session is None):
#             behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + "behavior_ms.csv"),sep=',')
#         else:
#             behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + session + "_" + "behavior_ms.csv"),sep=',')
#         data_types = ['RNFS', 'ALP', 'IALP', 'Time Stamp (ms)']
#         self.data = {}
#         for dt in data_types:            
#             if dt in behavior_data:
#                 self.data[dt] = behavior_data[dt]
#             else:
#                 print("No %s data found in minian file" % (dt))
#                 self.data[dt] = None

#         minian_path = os.path.join(dpath, "minian")
#         data = open_minian(minian_path)
#         data_types = ['A', 'C', 'S', 'E']
#         for dt in data_types:            
#             if dt in data:
#                 self.data[dt] = data[dt]
#             else:
#                 print("No %s data found in minian file" % (dt))
#                 self.data[dt] = None
        
#         self.data['unit_ids'] = self.data['C'].coords['unit_id'].values
#         self.dpath = dpath
#         self.data['collapsed_E'] = None

#         output_dpath = "/N/project/Cortical_Calcium_Image/analysis"
#         if session is None:
#             self.output_path = os.path.join(output_dpath, mouseID,day)
#         else:
#             self.output_path = os.path.join(output_dpath, mouseID,day,session)

#         if(os.path.exists(self.output_path) == False):
#             os.makedirs(self.output_path)

#     def total_calcium_events(self, unit: int):
#         """
#         Calculate the total number of calcium events for a given unit.
#         """
#         return self.data['E'].sel(unit_id=unit).max().values.item()

#     def get_timestep(self, type: str):
#         """
#         Return a list that contains contains the a list of the frames where
#         the ALP occurs
#         """
#         return np.flatnonzero(self.data[type])

#     def get_section(self, starting_frame: int, duration: float, delay: float = 0.0, include_prior: bool = False, type: str = "C") -> xr.Dataset:
#         """
#         Return the selection of the data that is within the given time frame.
#         duration indicates the number of frames.
#         """
#         # duration is in seconds convert to ms
#         duration *= 1000
#         delay *= 1000
#         start = self.data['Time Stamp (ms)'][starting_frame]
#         max_length = len(self.data['Time Stamp (ms)'])
#         if delay > 0:
#             frame_gap = 1
#             while self.data['Time Stamp (ms)'][starting_frame + frame_gap] - self.data['Time Stamp (ms)'][starting_frame] < delay:
#                 frame_gap += 1
#             starting_frame += frame_gap
#         if include_prior:
#             frame_gap = -1
#             while self.data['Time Stamp (ms)'][starting_frame] - self.data['Time Stamp (ms)'][starting_frame + frame_gap] < duration and starting_frame + frame_gap > 0:
#                 frame_gap -= 1
#             starting_frame += frame_gap
#             duration *= 2
#         frame_gap = 1
#         while self.data['Time Stamp (ms)'][starting_frame + frame_gap] - self.data['Time Stamp (ms)'][starting_frame] < duration and starting_frame + frame_gap < max_length:
#             frame_gap += 1


#         if type in self.data:
#             return self.data[type].sel(frame=slice(starting_frame, starting_frame+frame_gap))
#         else:
#             print("No %s data found in minian file" % (type))
#             return None
    
#     def get_AUC(self, section: xr.Dataset, section_event: xr.Dataset):
#         """
#         Calculate the area under the curve for a given section. Across all cells
#         """
#         if section.name != "S":
#             print("Invalid section type. Please use S not %s" % (section.name))
#             return None

#         amplitudes = self.get_amplitude(section, section_event)
#         total_auc = {}
#         for name, cell_events in amplitudes.items():
#             total_auc[name] = 0
#             for event_name, auc in cell_events.items():
#                 total_auc[name] += auc

#         return total_auc
    
#     def get_amplitude(self, section_signal: xr.Dataset, section_event: xr.Dataset):
#         """
#         Calculate the amplitude of the calcium event for a given section. Across all cells
#         """
#         if section_signal.name != "S":
#             print("Invalid section type. Please use S not %s" % (section_signal.name))
#             return None
#         if section_event.name != "E":
#             print("Invalid section type. Please use S not %s" % (section_event.name))
#             return None

#         all_cell_amplitudes = {}

#         for unit_id in self.data['unit_ids']:
#             cell_amplitudes = {}
#             signal = section_signal.sel(unit_id=unit_id).values
#             event = section_event.sel(unit_id=unit_id).values
#             unique_events = np.unique(event)
#             for event_id in unique_events:
#                 if event_id == 0:
#                     continue
#                 cell_amplitudes[event_id] = np.sum(signal[event == event_id])
#             all_cell_amplitudes[unit_id] = cell_amplitudes
        
#         return all_cell_amplitudes
    
#     def get_frequency(self, section: xr.Dataset, time: float):
#         """
#         Calculate the frequency of the calcium events for a given section. Across all cells
#         """
#         if section.name != "E":
#             print("Invalid section type. Please use S not %s" % (section.name))
#             return None

#         all_cell_frequency = {}

#         for unit_id in self.data['unit_ids']:
#             cell_frequency = {}
#             event = section.sel(unit_id=unit_id).values
#             unique_events = np.unique(event)
#             all_cell_frequency[unit_id] = len(unique_events)-1 / time
#         return all_cell_frequency

    
#     def count_events(self, a: np.ndarray) -> np.ndarray:
#         """
#         count the number of events in a given array.
#         We do -1 to compensate for the 0 in the array
#         """
#         return np.unique(a).size - 1

        
#     def collapse_E_events(self, smoothing="gauss", kwargs=None) -> None:
#         """
#         Collapse the E values by summing up the values.
#         """
#         non_collapsed_E = xr.apply_ufunc(
#             self.normalize_events,
#             self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
#             input_core_dims=[["frame"]],
#             output_core_dims=[["frame"]],
#             dask="parallelized",
#             output_dtypes=[self.data['E'].dtype],
#         )

#         self.data['collapsed_E'] = non_collapsed_E.sum(dim='unit_id')

#         if smoothing == "gauss":
#             self.data['collapsed_E'] = xr.apply_ufunc(
#                 gaussian_filter1d,
#                 self.data['collapsed_E'],
#                 input_core_dims=[["frame"]],
#                 output_core_dims=[["frame"]],
#                 dask="parallelized",
#                 kwargs=kwargs,
#                 output_dtypes=[self.data['E'].dtype]
#             )
#         elif smoothing == "mean":
#             self.data['collapsed_E'] = xr.apply_ufunc(
#                 self.moving_average,
#                 self.data['collapsed_E'],
#                 input_core_dims=[["frame"]],
#                 output_core_dims=[["frame"]],
#                 dask="parallelized",
#                 kwargs=kwargs,
#                 output_dtypes=[self.data['E'].dtype]
#             ).compute()
    
#     def collapse_E_events_AUC(self) -> None:
#         """
#         Collapse the E values by summing up the values.
#         """
#         non_collapsed_E = xr.apply_ufunc(
#             self.normalize_events,
#             self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
#             input_core_dims=[["frame"]],
#             output_core_dims=[["frame"]],
#             dask="parallelized",
#             output_dtypes=[self.data['E'].dtype],
#         )

#         non_collapsed_E *= self.data['S']

#         self.data['collapsed_E_AUC'] = non_collapsed_E.sum(dim='unit_id')

#     def moving_average(self, x, w=100, type='constant'):
#         return np.convolve(x, np.ones(w), type) / w

#     def collapse_E_events_peak(self) -> None:
#         '''
#         Get the events' peak
#         '''
#         non_collapsed_E_peak = xr.apply_ufunc(
#             self.derivative,
#             self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
#             input_core_dims=[["frame"]],
#             output_core_dims=[["frame"]],
#             dask="parallelized",
#             output_dtypes=[self.data['E'].dtype],
#         ).compute()
#         self.data['collapsed_E_peak'] = non_collapsed_E_peak.sum(dim='unit_id')

#     def derivative(self, a: np.ndarray) -> np.ndarray:
#         a = a.copy()
#         b = np.roll(a, 1, axis=1)
#         # b[:, 0] = 0
#         c = a - b
#         c[c > 0] = 0
#         c[c < 0] = 1
#         c = np.roll(c, -1, axis=1)
#         return c



#     def find_events_peak(self, a: np.ndarray) -> np.ndarray:
#         a = a.copy()
#         print(a.shape)
#         res = np.zeros(np.shape(a))
#         for row,b in enumerate(a):
#             u,i,c=np.unique(b,return_index = True,return_counts = True)
#             for m in range(1,len(u)):                
#                 res[row,(i[m]+c[m]-1)]=u[m]
#         return res


#     def normalize_events(self, a: np.ndarray) -> np.ndarray:
#         """
#         All positive values are converted to 1.
#         """
#         a = a.copy()
#         a[a > 0] = 1
#         return a

    
#     def centroid(self, verbose=False) -> pd.DataFrame:
#         """
#         Compute centroids of spatial footprint of each cell.

#         Parameters
#         ----------
#         A : xr.DataArray
#             Input spatial footprints.
#         verbose : bool, optional
#             Whether to print message and progress bar. By default `False`.

#         Returns
#         -------
#         cents_df : pd.DataFrame
#             Centroid of spatial footprints for each cell. Has columns "unit_id",
#             "height", "width" and any other additional metadata dimension.
#         """
#         A = self.data['A']
#         def rel_cent(im):
#             im_nan = np.isnan(im)
#             if im_nan.all():
#                 return np.array([np.nan, np.nan])
#             if im_nan.any():
#                 im = np.nan_to_num(im)
#             cent = np.array(center_of_mass(im))
#             return cent / im.shape

#         gu_rel_cent = da.gufunc(
#             rel_cent,
#             signature="(h,w)->(d)",
#             output_dtypes=float,
#             output_sizes=dict(d=2),
#             vectorize=True,
#         )
#         cents = xr.apply_ufunc(
#             gu_rel_cent,
#             A.chunk(dict(height=-1, width=-1)),
#             input_core_dims=[["height", "width"]],
#             output_core_dims=[["dim"]],
#             dask="allowed",
#         ).assign_coords(dim=["height", "width"])
#         if verbose:
#             print("computing centroids")
#             with ProgressBar():
#                 cents = cents.compute()
#         cents_df = (
#             cents.rename("cents")
#             .to_series()
#             .dropna()
#             .unstack("dim")
#             .rename_axis(None, axis="columns")
#             .reset_index()
#         )
#         h_rg = (A.coords["height"].min().values, A.coords["height"].max().values)
#         w_rg = (A.coords["width"].min().values, A.coords["width"].max().values)
#         cents_df["height"] = cents_df["height"] * (h_rg[1] - h_rg[0]) + h_rg[0]
#         cents_df["width"] = cents_df["width"] * (w_rg[1] - w_rg[0]) + w_rg[0]
#         return cents_df

#     def get_filted_C(self) -> None:
#         non_collapsed_E = xr.apply_ufunc(
#             self.normalize_events,
#             self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
#             input_core_dims=[["frame"]],
#             output_core_dims=[["frame"]],
#             dask="parallelized",
#             output_dtypes=[self.data['E'].dtype],
#         )
#         filted_C = self.data['C'] * non_collapsed_E
#         self.data['filted_C'] = filted_C

#     def smoothed_C(self,window_length = 6, n = 3,mode = "savgol",unit_id = []) -> None:
#         self.data['smoothed_C'] = self.data['C']
#         if unit_id is None:
#             unit_id = self.data['unit_ids']
#         if mode =="savgol":
#             smoothed_C = xr.apply_ufunc(
#                 savgol_filter,
#                 self.data['C'].sel(unit_id=unit_id),
#                 window_length,
#                 n,
#                 input_core_dims=[["frame"],[],[]],
#                 output_core_dims=[["frame"]],
#                 dask="parallelized",
#                 output_dtypes=[self.data['C'].dtype],
#             )
#         elif mode =="gauss":
#             smoothed_C = xr.apply_ufunc(
#                 gaussian_filter1d,
#                 self.data['C'].sel(unit_id=unit_id),
#                 3,
#                 input_core_dims=[["frame"],[]],
#                 output_core_dims=[["frame"]],
#                 dask="parallelized",
#                 output_dtypes=[self.data['C'].dtype],
#             )
#         for uid in unit_id:
#             self.data['smoothed_C'].sel(unit_id = uid).values = smoothed_C.sel(unit_id=uid)

#     def smoothed_filted_C(self) -> None:
#         non_collapsed_E = xr.apply_ufunc(
#             self.normalize_events,
#             self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
#             input_core_dims=[["frame"]],
#             output_core_dims=[["frame"]],
#             dask="parallelized",
#             output_dtypes=[self.data['E'].dtype],
#         )
#         smoothed_filted_C = self.data['smoothed_C'] * non_collapsed_E
#         self.data['smoothed_filted_C'] = smoothed_filted_C

# class Features:
#     '''
#         Parameters
#         ----------
#         ALP
#     '''

#     def __init__(self,
#                 A: dict,
#                 ALP: List[xr.DataArray],
#                 IALP: List[xr.DataArray],
#                 RNFS: List[xr.DataArray],  
#                 events: Optional[List[str]] = None, 
#                 description: Optional[str] = None,
#                 dist_met: Optional[str] = None, 
#         ):
#         self.A = A
#         self.ALPlist = ALP
#         self.IALPlist = IALP
#         self.RNFSlist = RNFS
#         self.events = events
#         self.dist_met = dist_met
#         self.description = description
#         self.set_vector(self.events)

    
#     def set_timefilter(self):
#         ALP = {}
#         IALP = {}
#         RNFS = {}
#         if self.timefilter is None:
#             for i in self.ALPlist:
#                 for j in i.coords['unit_id'].values:
#                     try:
#                         ALP[j]
#                     except:
#                         ALP[j] = np.array([])
#                     ALP[j] = np.r_['-1', ALP[j], np.array(i.sel(unit_id=j).values)]
#             for i in self.IALPlist:
#                 for j in i.coords['unit_id'].values:
#                     try:
#                         IALP[j]
#                     except:
#                         IALP[j] = np.array([])
#                     IALP[j] = np.r_['-1', IALP[j], np.array(i.sel(unit_id=j).values)]
#             for i in self.RNFSlist:
#                 for j in i.coords['unit_id'].values:
#                     try:
#                         RNFS[j]
#                     except:
#                         RNFS[j] = np.array([])
#                     RNFS[j] = np.r_['-1', RNFS[j], np.array(i.sel(unit_id=j).values)]
#         self.ALP = ALP
#         self.IALP = IALP
#         self.RNFS = RNFS
#         self.set_vector(self.events)


#     def set_events(self, events:List[str]):
#         self.events = events
#         self.set_vector(self.events)

#     def set_vector(self, events:list):
#         '''
#         event :  str, list
#             event can be ALP/IALP/RNFS
#         '''
#         if events is None:
#             events=['ALP','IALP','RNFS']
#         values = {}
#         if 'ALP' in events:
#             for key in self.ALP:
#                 try:
#                     values[key]
#                 except:
#                     values[key] = np.array([])
#                 values[key] = np.r_['-1', values[key], self.ALP[key]]            
#         if 'IALP' in events:
#             for key in self.IALP:
#                 try:
#                     values[key]
#                 except:
#                     values[key] = np.array([])
#                 values[key] = np.r_['-1', values[key], self.IALP[key]]
#         if 'RNFS' in events:
#             for key in self.RNFS:
#                 try:
#                     values[key]
#                 except:
#                     values[key] = np.array([])
#                 values[key] = np.r_['-1', values[key], self.RNFS[key]]
#         self.values = values
    
#     def set_description(self, content:str):
#         self.description = content

#     def reset_dataArray(self, a: np.ndarray):
#         a = np.array([])
#         return a

