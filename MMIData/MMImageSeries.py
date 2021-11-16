# from global_imports import *
from pathlib import Path
import numpy as np
import skimage as ski
from skimage import io


# class MMIFile(Path):


# 	def __init__(self, path_str):
# 		super().__init__(self, path_str)

# 		FOV_str, t_str, self.channel = fpath.stem.split("_", 2)
# 		self.FOV = int(FOV_str[2:])
# 		self.t = int(t[1:])

# 	_flavour = Path("")._flavour


# 	def load(self):
# 		return ski.io.imread(self)

# 	def metadata(self):
# 		yield self.FOV
# 		yield self.t
# 		return self.channel

class MMIFileSeries():

	def __init__(self, folder_paths, dtypes, set_dims=None, raise_if_none_found=True):
		''' 
		set_dims: (n_timesteps, n_FOVs, ("channel1", "channel2", ...)) if not None
		'''
		self.raise_if_none_found = raise_if_none_found
		self._sort_out_input_formats(folder_paths, dtypes)

		self._construct_map()

		if set_dims is None:
			self._parse_ranges()
		# elif isinstance(set_dims, MMIFileSeries):
			# self.n_timesteps = set_dims.n_timesteps
			# self.n_FOVs = set_dims.n_channels
			# self.channel_names = 
		else:
			self.n_timesteps, self.n_FOVs, self.channel_names = set_dims
			self.channel_names = list(self.channel_names)
			self.n_channels = len(self.channel_names)

		self._construct_ndarray()


	def reload(self):
		self._construct_map()
		self._construct_ndarray()


	def _sort_out_input_formats(self, folder_paths, dtypes):

		self.folder_paths = folder_paths
		self.dtypes = dtypes

		if type(self.folder_paths) not in (list, tuple):
			self.folder_paths = [self.folder_paths]
		if type(self.dtypes) not in (list, tuple):
			self.dtypes = [self.dtypes]

		for i, folder_path in enumerate(self.folder_paths):
			self.folder_paths[i] = Path(folder_path)


	def _construct_map(self):

		self.fpaths_map = []

		for folder_path in self.folder_paths:
			for dtype in self.dtypes:
				self.fpaths_map += [fp for fp in folder_path.glob(f"*.{dtype}")]

		if len(self.fpaths_map) == 0 and self.raise_if_none_found:
			raise Exception("No files found!")


	def _parse_ranges(self):

		max_t = 0
		max_FOV = 0
		channels = []

		xy_t_ch = [fpath.stem.split("_", 2) for fpath in self.fpaths_map]

		for xy, t, ch in xy_t_ch:

			t_int = int(t[1:])
			max_t = max(max_t, t_int)
			FOV_int = int(xy[2:])
			max_FOV = max(max_FOV, FOV_int)

			if ch not in channels:
				channels.append(ch)

		channels.sort()
		self.channel_names = channels

		self.n_timesteps = max_t + 1

		self.n_FOVs = max_FOV + 1

		self.n_channels = len(self.channel_names)


	# def add_channels(*channel_names):
	# 	self.channel_names = self.channel_names + channel_names
	# 	self.n_channels = len(self.channel_names)

	# 	self._construct_map()



	def _construct_ndarray(self):

		self.img_fpath_ids = -1 + np.zeros((self.n_timesteps, self.n_FOVs, self.n_channels), dtype=int)

		for i, fpath in enumerate(self.fpaths_map):
			xy, t, ch = fpath.stem.split("_", 2)
			t = int(t[1:])
			FOV = int(xy[2:])
			try:
				ch = self.channel_names.index(ch)
			except:
				raise KeyError("This channel has not been added to the MMIFile data structure")

			self.img_fpath_ids[t,FOV,ch] = i



	def __getitem__(self, s_obj):

		if len(s_obj) > 2 and type(s_obj[-1]) == str:
			s_obj = np.s_[s_obj[0], s_obj[1], self.channel_names.index(s_obj[-1])]

		# if len(s_obj) > 2 and type(s_obj[-1]) in (tuple, list):
		# 	last_part = list(s_obj[-1])
		# 	for i, elem in enumerate(last_part):
		# 		if type(elem) == str:
		# 			last_part[i] = self.channel_names.index(elem)
		# 	s_obj = np.s_[s_obj[0], s_obj[1], tuple(last_part)]


		def map_fpath(idx):
			return (self.fpaths_map + [None])[idx]

		res = np.vectorize(map_fpath)(self.img_fpath_ids[s_obj])
		if res.size == 1:
			return res[()]
		else:
			return res 


	def PC_img_fpath(self, timestep, FOV):
		return self[timestep, FOV, 0]

	def FL_img_fpath(self, timestep, FOV, channel):
		return self[timestep, FOV, 1 + channel]



class MMImageSeries(MMIFileSeries):

	def __init__(self, folder_paths, dtypes="png", raise_if_none_found=True):

		self.raise_if_none_found = raise_if_none_found
		self._sort_out_input_formats(folder_paths, dtypes)

		self._construct_map()

		print(self.fpaths_map)

		self._parse_ranges()

		if "PC" in self.channel_names:
			# Make it the first index.
			idx = self.channel_names.index("PC")
			del self.channel_names[idx]
			self.channel_names = ["PC"] + self.channel_names

		self._construct_ndarray()

