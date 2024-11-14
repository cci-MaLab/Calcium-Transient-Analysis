Before you Start
================

This project was built in mind with integrating `Minian <https://github.com/denisecailab/minian>`_
output files. Below will be two sections delineating what to do if you're coming from a Minian
project and a more comprehensive guide if you come from a different CNMF related project.

.. _non-minian projects:

Non-Minian Projects
-------------------

If your project is based on non-Minian approach, you can look at the delineation of data structures
below, so you can reach parity. Since there a multitude of different approaches to store CNMF related
data, we will only cover what the end result should look like.

Following the approach in Minian, the data should be stored in a `zarr <https://zarr.readthedocs.io/en/stable/>`_ format,
due to their usage of `xarray <http://xarray.pydata.org/en/stable/>`_. Xarray is a numpyesque library that allows for efficient
memory usage and parallelization. The following data is expected:

- `A.zarr`: A 3D array of shape `(unit_id, height, width)`. This array represents the spatial footprints of the neurons.
- `C.zarr`: A 2D array of shape `(unit_id, frame)`. This array represents the calcium traces of the neurons.
- `S.zarr`: A 2D array of shape `(unit_id, frame)`. This array represents the spike/firing rate.
- `YrA.zarr`: A 2D array of shape `(unit_id, frame)`. This array represents the raw signals/residual traces.
- `DFF.zarr`: A 2D array of shape `(unit_id, frame)`. This array represents the deltaF/F traces.

In the case that you don't have a DFF trace, our project will generate and calculate it for you, using the `detrend_df_f` function from `CaImAn <https://github.com/flatironinstitute/CaImAn/blob/bb55800806f0898592d79dcc705a0b53ccd01ec3/caiman/source_extraction/cnmf/utilities.py#L442>`_.

However, it is heavily encouraged to have your own DFF trace calculated, as it will be more accurate and tailored to your data.
The calculation of DFF if initially omitted will require the following additional data:

- `f.zarr`: A 1D array of shape `(frame)`. Estimation of background flourescence at each frame.
- `b.zarr`: A 2D array of shape `(height, width)`. Spatial background component.

.. _numpy to xarray:

How to convert numpy to xarray
______________________________

If you have your data in numpy format, you can convert it to an xarray using the following code snippet:

.. code-block:: python

    import xarray as xr
    import numpy as np

    unit_ids = np.random.randint(0, 100, 100)

    # Create a numpy array
    C_numpy = np.random.rand(100, 100)

    # Create an xarray
    C = xr.DataArray(
        C_numpy,
        dims=["unit_id", "frame"],
        coords=dict(
            unit_id=unit_ids,
            frame=np.arange(C_numpy.shape[1]),
        ),
        name="C"
    )

    # Save the xarray to a zarr file
    C.to_zarr("C.zarr")

    # Now for A.zarr
    A_numpy = np.random.rand(100, 100, 100)
    A = xr.DataArray(
        A_numpy,
        dims=["unit_id", "height", "width"],
        coords=dict(
            unit_id=unit_ids,
            height=np.arange(A_numpy.shape[1]),
            width=np.arange(A_numpy.shape[2]),
        ),
        name="A"
    )

    A.to_zarr("A.zarr")

Repeat the process above for other variables.

.. _video files:

Video files
___________

The following videos are expected to likewise be in a zarr format:

 - `varr.zarr`: A 3D array of shape `(frame, height, width)`. This array represents the raw video data. When chunked, ensure that it is chunked along the frame axis and not the height or width.
 - `Y_fm_chk.zarr`: A 3D array of shape `(frame, height, width)`. This array represents the processed video data. When chunked, ensure that it is chunked along the frame axis and not the height or width.
 - `Y_hw_chk.zarr`: A 3D array of shape `(frame, height, width)`. This array represents the processed video data. When chunked, ensure that it the frames are intact and it is chunked along heigh and width. It will be automatically created if it doesn't exist (However it will take a considerable amount of time).
 - (Optional) `behavior_video.zarr`: A 3D array of shape `(frame, height, width)`. This array represents the behavior video data. When chunked, ensure that it is chunked along the frame axis and not the height or width.

We are aware that the recording framerate in the behavior video will most likely differ to that of the calcium imaging video.
We account for that in our project, you need to ensure that the first and last frame of the behavior video, roughly aligns with the first and last frame of the calcium imaging video.

Chunking is an important aspect of zarr, it dictates in what way the data is stored on disk and how it is read into memory.
For the purposes of efficient GUI usage, you should chunk the data as stated above. To load in your data into a proper format
and to have it chunked correctly, you can follow the steps in the `Minian documentation <https://minian.readthedocs.io/en/stable/pipeline/notebook_2.html>`_.

Once you have your data in the correct format, you can proceed to the `Minian` section below.

.. _minian projects:

Minian Projects
---------------

Loading in your data will require 2 folders and a csv file:

- `data`: This folder should contain the following files:
    - `A.zarr`
    - `C.zarr`
    - `S.zarr`
    - `YrA.zarr`
    - `DFF.zarr` (In the case that you don't have this, include `f.zarr` and `b.zarr` so it will be calculated for you)
- `videos`: This folder should contain the following files:
    - `varr.zarr`
    - `Y_fm_chk.zarr`
    - `Y_hw_chk.zarr` (Optional, will be created if it doesn't exist)
    - `behavior_video.zarr` (Optional, look at the `video files`_ section for more information)
- `behavior.csv`: This file contains both millisecond time information as well as the behavior data, where 0 represents no event occurred and 1 represents that an event happened. The following indicates the column information:
    - `Frame Number`: The frame number of the video
    - `Time Stamp (ms)`: The time in milliseconds
    - (Optional) `RNF`: Reinforcement
    - (Optional) `ALP`: Active lever press
    - (Optional) `ILP`: Inactive lever press
    - (Optional) `ALP_Timeout`: Active lever press timeout

The following is an example of what the csv file could look like:

.. list-table:: Example CSV File
   :header-rows: 1

   * - Frame Number
     - Time Stamp (ms)
     - RNF
     - ALP
     - ILP
     - ALP_Timeout
   * - 0
     - 0
     - 0
     - 0
     - 0
     - 0
   * - 1
     - 33
     - 0
     - 0
     - 0
     - 0
   * - 2
     - 66
     - 0
     - 0
     - 0
     - 0
   * - 3
     - 100
     - 0
     - 0
     - 0
     - 0
   * - 4
     - 133
     - 0
     - 0
     - 0
     - 0
   * - 5
     - 166
     - 0
     - 0
     - 0
     - 0
   * - 6
     - 200
     - 0
     - 0
     - 0
     - 0
   * - 7
     - 233
     - 0
     - 0
     - 0
     - 0
   * - 8
     - 266
     - 0
     - 0
     - 0
     - 0
   * - 9
     - 300
     - 0
     - 0
     - 0
     - 0
   * - 10
     - 333
     - 0
     - 0
     - 0
     - 0
   * - 11
     - 366
     - 1
     - 0
     - 0
     - 0
   * - 12
     - 400
     - 0
     - 0
     - 0
     - 0

Creating the Config File
------------------------

The final step is to create a config.ini file that will tell the GUI where to find the necessary data.
Below is a template that you can adjust to your needs:

.. code-block:: ini

    [Session_Info]
    mouseid = AA058
    day = D1
    session = S4
    group = None
    data_path = C:\path\to\folder\that\contains\data
    video_path = C:\path\to\folder\that\contains\videos
    behavior_path = C:\path\to\folder\that\contains\behavior.csv