from .imports import *
from ._trench_signal import Trench, offset_TBLR_slice


class SegmentedMMI():
	'''
	Data-efficient storage of a trench segmentation.
	'''

	def __init__(self, img_fpaths, mask_folder_path, labelled_cell_imgs, trenches, rot_angle, save_on_init=True):

		self.img_fpaths = img_fpaths
		self.PC_img_fpath = img_fpaths[0]
		self.mask_folder_path = mask_folder_path

		fp1, fp2 = SegmentedMMI.get_save_fpaths(self.PC_img_fpath, mask_folder_path)
		self.save_img_fpath, self.save_pickle_fpath = fp1, fp2

		self.trenches = trenches
		self.trench_n_cells = [img.max() for img in labelled_cell_imgs]

		self.FL_bg_means, self.FL_bg_stds = self.get_FL_bg_vals()

		self.n_imgs = len(trenches)
		self.rot_angle = rot_angle

		self._save(labelled_cell_imgs)



	def get_trench_ids(self):
		return [round((trench.TBLR[2]*2)/155) for trench in self.trenches]



	def get_trench_idx_by_id(self, id_):
		trench_ids = self.get_trench_ids()
		print(id_, trench_ids)
		if id_ in trench_ids:
			return trench_ids.index(id_)
		else:
			return None


	def get_FL_bg_vals(self):
		''' This still includes a little bit of light from the cells '''

		means = []
		stds = []
		for fpath in self.img_fpaths[1:]:
			img = ski.io.imread(fpath)
			msk = np.ones_like(img, dtype=bool)

			for trench in self.trenches:
				msk[trench.slice] = False

			msk = msk[::4,::4]
			img = img[::4,::4]

			msk[:100,:] = False

			struct_elem = np.ones((9, 9))
			struct_elem[:,:2] = 0
			struct_elem[:,-2:] = 0

			msk1 = ski.morphology.erosion(msk, struct_elem)
			msk1 = ski.morphology.erosion(msk1, struct_elem)
			msk1 = ski.morphology.erosion(msk1, struct_elem)

			''''hack. Todo: fix'''
			means.append(img[msk].mean())
			stds.append(img[msk].std())
			# print(img[msk].max(), img[msk].min(), img[msk].mean(), img[msk].std())

			# img[msk != True] = 0

			# plt.imshow(img)
			# plt.show()
			# img[msk1 != True] = 0
			# plt.imshow(img)
			# plt.show()

		return means, stds


	@classmethod
	def create_with_handle(cls, *args, **kwargs):
		new_obj = cls(*args, **kwargs)
		return new_obj, new_obj.save_pickle_fpath

	@classmethod
	def load(cls, pickle_fpath):
		with open(pickle_fpath, "rb") as f:
			loaded = pickle.load(f)
		return loaded

	@staticmethod
	def get_save_fpaths(PC_img_fpath, mask_folder_path):

		save_img_fname = PC_img_fpath.stem[:-3] + "_segmented.png"
		save_pickle_fname = PC_img_fpath.stem[:-3] + "_segmented_meta.pickle"

		return mask_folder_path / save_img_fname, mask_folder_path / save_pickle_fname



	def get_rot_fn(self):
		def rot_fn(img):
			return ski.transform.rotate(img, self.rot_angle, clip=False, preserve_range=True)
		return rot_fn



	def _save(self, labelled_cell_imgs):
		ski.io.imsave(self.save_img_fpath, np.hstack(labelled_cell_imgs).astype(np.ubyte))
		with open(self.save_pickle_fpath, "wb") as f:
			pickle.dump(self, f)




	def load_masks(self, dilate=None):
		img = ski.io.imread(self.save_img_fpath)
		if dilate is not None:
			return np.array([ski.segmentation.expand_labels(img1, dilate) for img1 in np.hsplit(img, self.n_imgs)])
		else:
			return np.hsplit(img, self.n_imgs)


	def load_masked_PC(self):
		return self._load_masked_data(self.img_fpaths[0])

	def load_masked_FL(self, channel):
		return self._load_masked_data(self.img_fpaths[1+channel])

	def load_masked_channel(self, channel, dilate=None, offset=None):
		return self._load_masked_data(self.img_fpaths[channel], dilate=dilate, offset=offset)

	def load_all_masked_data(self):
		return self._load_masked_data(self.img_fpaths)

	def load_all_masked_FL_channels(self, dilate=None):
		return self._load_masked_data(self.img_fpaths[1:], dilate=dilate)


	# def mask_img(self, img):
		# overall_mask = 
		# for i, trench in enumerate(self.trenches):
	def mask_out_trench(self, img, trench, cell_mask, offset=None):
		if offset is None:
			slc = trench.central_slice
		else:
			slc = offset_TBLR_slice(trench.central_TBLR, offset)

		img1 = img[slc]
		img1[cell_mask==0] = 0

		return img1
		

	def _load_masked_data(self, *fpaths, dilate=None, offset=None):

		cell_masks = self.load_masks(dilate)
		# if dilate is not None:
		#	 cell_masks = [ski.segmentation.expand_labels(cm, dilate) for cm in cell_masks]

		# plt.imshow(cell_masks[0].T)
		# plt.show()

		if len(fpaths) == 1:
			fpath = fpaths[0]
			# print(ski.io.imread(fpath).max())
			rotated_img = self.get_rot_fn()(ski.io.imread(fpath))
			# print(rotated_img.max())

			# plt.imshow(rotated_img)
			# plt.show()

			return [
				self.mask_out_trench(rotated_img, trench, cell_masks[i], offset=offset) 
					for i, trench in enumerate(self.trenches)
			]
		else:
			to_return = [[]*len(fpaths)]
			for j, fpath in enumerate(fpaths):

				rotated_img = self.get_rot_fn()(ski.io.imread(fpath))

				to_return[j] = [
					# np.bitwise_and(rotated_img[trench.central_slice], cell_masks[i]) 
					self.mask_out_trench(rotated_img, trench, cell_masks[i], offset=offset) 
						for i, trench in enumerate(self.trenches)
				]

			return to_return
