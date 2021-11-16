from .imports import *
from .image_utils import rescale_to_01
from .segmentation import segment_cells_in_trench
from .data_processing import get_rot_angle, get_rot_transform
from .MMImageSeries import MMImageSeries, MMIFileSeries
from .SegmentedMMI import SegmentedMMI
from ._trench_signal import TrenchSignalDetector, Trench

class MMIData():

	def __init__(self, data_path, img_folder_name, img_dtype="png"):

		self.data_path = Path(data_path)

		self.imgs = MMImageSeries(self.data_path / img_folder_name, img_dtype)
		self.TSD = TrenchSignalDetector()

		self.n_FOVs = self.imgs.n_FOVs
		self.n_timesteps = self.imgs.n_timesteps

		self.mask_folder_path = self.data_path / (img_folder_name + "_masks")
		self.mask_folder_path.mkdir(parents=True, exist_ok=True)

		self.seg = MMIFileSeries(self.mask_folder_path, ("png", "pickle"), 
			set_dims=(self.n_timesteps, self.n_FOVs, ("segmented", "segmented_meta")),
			raise_if_none_found=False
			)


	def segment_all(self):

		for FOV in range(0, Data.n_FOVs):
			for t in range(0, Data.n_timesteps):

				print(f"Timestep {t}, FOV {FOV}")
				
				Data.segment_single(t, FOV)


	def segment_single(self, timestep, FOV, skip_if_already_segmented=True, show_on_error=True):

		t = timestep

		PC_fpath = self.imgs[t, FOV, "PC"]

		if PC_fpath is None:
			return None

		if skip_if_already_segmented:
			if self.seg[t,FOV,"segmented_meta"] is not None:
				print("Already segmented this image! trying to load...")
				try:
					loaded = SegmentedMMI.load(self.seg[t,FOV,"segmented_meta"])
				except Exception as e:
					print(f"Failed: \n{e}")

				return loaded

		PC_img = ski.io.imread(PC_fpath)

		rot_angle = get_rot_angle(PC_img, downscale_f=0.25, extra_precise=True, print_progress=False)
		rot_fn = get_rot_transform(rot_angle)

		PC_img = rot_fn(PC_img)

		# print(self.imgs.PC_img_fpath(t, FOV))
		# plt.imshow(PC_img)
		# plt.show()

		try:
			# This does not copy data, just gives slices of PC_img
			trenches = self.TSD.extract_trenches(PC_img, show_img=False)
		except Exception as e:
			print("Failed with error:")
			print(e)
			if show_on_error:
				plt.imshow(PC_img)
				plt.show()
			return None
			
			# try:
			#     trenches = self.TSD.extract_trenches(PC_img, show_img=True)
			# except:
			#     pass


		# cell_imgs are cropped ubyte trench images with cells
		# masked by labelled integers.
		cell_imgs = []

		# print("starting segmentation...")

		for i, trench in enumerate(trenches):
			# should enforce these to have the same dimensions.
			cell_imgs.append(segment_cells_in_trench(PC_img, trench).T)

		# print("finished segmentation.")



		if False:
			trench_central_imgs = [PC_img[trench.central_slice] for trench in trenches]

			# plt.imshow(np.vstack([x.T for x in trench_central_imgs]))
			# plt.show()
			# plt.imshow(np.vstack([x.T for x in cell_imgs]))
			# plt.show()
			for i in range(len(cell_imgs)):
				trench_central_imgs[i][cell_imgs[i]==0] = 0
			plt.imshow(np.vstack([x.T for x in trench_central_imgs]))
			plt.show()



		saved_segmentation, fpath = SegmentedMMI.create_with_handle(
			self.imgs[t,FOV,:], self.mask_folder_path, cell_imgs, trenches, rot_angle
		)

		self.seg.reload()

		return saved_segmentation


	# def load_segmented(self, t, FOV):
		# PC_img_fpath = self.imgs[t,FOV,"PC"]
		# mask_folder_path = ...
		# SegmentedMMI.load(mask_folder_path)

