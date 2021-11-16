from .imports import *
from .image_utils import *



def segment_cells_in_trench(rotated_PC_img, trench):

	labelled, local_min_coords = labelled_cell_masks(rotated_PC_img[trench.central_slice].T, False)
	return labelled



def last_flood_before_edge_hit(img, coords, n_flood_intervals, n_back=1):
	'''
	Improvements to make:
	- search t-space in a more intelligent way 
	- go back until...........
	'''

	x_radius = 50
	min_x = max(coords[1] - x_radius, 0)
	max_x = min(coords[1] + x_radius, img.shape[1]-1)

	img = img_as_ubyte(img)

	to_flood = img[:,min_x:max_x].copy()
	last_flooded = np.zeros_like(to_flood)

	ts = np.linspace(img.min(), img.max(), n_flood_intervals)
	for it, t in enumerate(ts):
		cy, cx = coords
		flooded = ski.segmentation.flood(to_flood, (cy, cx-min_x), tolerance=int(t))
		# plt.imshow(flooded)
		# plt.show()top`

		if np.any(flooded[-2:,:] == 1) or np.any(flooded[:2,:] == 1) or\
			np.any(flooded[:,-1:] == 1) or np.any(flooded[:,:1] == 1):
			break


	if it < n_back:
		return np.zeros_like(img)
	else:
		to_return = np.zeros_like(img)
		to_return[:,min_x:max_x] = ski.segmentation.flood(to_flood, (cy, cx-min_x), tolerance=int(ts[it-n_back]))
		return to_return





def labelled_cell_masks(img, show_img=False):

	rescale_to_01_inplace(img)

	locality_radius = 51
	threshold = ski.filters.threshold_local(img, locality_radius)

	img_local_sub = img - threshold
	# img_cutoff = cutoff_n_sds_above_mean(img_local_sub)	

	small_blur = ski.filters.gaussian(img_local_sub, 2)

	rescale_to_01_inplace(small_blur)

	local_min_coords = get_local_maxima(-small_blur, show_img=show_img)
	local_min_coords = local_min_coords[(img.shape[0]*.25 < local_min_coords[:,0]) & (local_min_coords[:,0] < img.shape[0]*.75)]

	labelled = np.zeros_like(img).astype(bool)
	j = 1

	for coords in local_min_coords:
		last_flooded = last_flood_before_edge_hit(img, coords, 51, 1)
		# labelled = np.maximum(labelled, last_flooded*j)
		existing_labels = labelled[last_flooded!=0]

		max_of_existing_labels = 0
		if len(existing_labels) > 0:
			max_of_existing_labels = np.max(existing_labels)

		if max_of_existing_labels != 0:
			labelled = np.maximum(labelled, last_flooded*max_of_existing_labels)
		else:
			labelled = np.maximum(labelled, last_flooded*j)
			j += 1




	##########################################
	# TODO:
	# SHOULD SET ALL LABELS TO 1
	# THEN, LOOK AT SIZE DISTRIBUTION
	# THEN, CUT OFF ALL BELOW CERTAIN SIZE
	# THEN, ERODE
	# THEN, EXPAND_LABELS
	# NEED TO RE-LABEL TO ALLOW THIS

	# for j in range(1,labelled.max()+1):
		# if (labelled==j.sum() < 50:
			# labelled[labelled==j] = 0



	# labelled = ski.morphology.erosion(labelled, ski.morphology.disk(1))
	# labelled = ski.morphology.erosion(labelled, ski.morphology.disk(1))
	# labelled = ski.morphology.closing(labelled, ski.morphology.disk(1))
	# labelled = ski.morphology.opening(labelled, ski.morphology.disk(1))
	# labelled = ski.morphology.opening(labelled, ski.morphology.disk(2))
	# labelled = ski.morphology.opening(labelled, ski.morphology.disk(3))
	# labelled = ski.morphology.opening(labelled, ski.morphology.disk(4))
	# labelled = ski.morphology.opening(labelled, ski.morphology.disk(5))

	# labelled = ski.morphology.erosion(labelled, ski.morphology.disk(1))


	##################################
	# some filtering steps
	struct_elem = np.zeros((3,3))
	struct_elem[:,1] = 1

	labelled = ski.morphology.erosion(labelled, struct_elem)
	labelled = ski.morphology.dilation(labelled, struct_elem)




	old_n = labelled.max()
	labelled, n = ski.measure.label(labelled, return_num=True)

	to_remove = set()
	for label in range(1, n+1):
		if np.sum(labelled == label) < 80:
			to_remove.add(label)
		if np.any(labelled[:,:15] == label):
			to_remove.add(label)
		if np.any(labelled[:,-15:] == label):
			to_remove.add(label)

	for label in to_remove:
		labelled[labelled==label] = 0

	# masked = img.copy()
	# masked[labelled==0] = 0

	# plt.imshow(img)
	# plt.show()
	# plt.imshow(labelled)
	# plt.show()

	if show_img:
		plt.imshow(np.vstack(
			[(j - j.min())/j.max() for j in [
			img,
			threshold,
			img_local_sub,
			labelled,
			masked
			# rescale_to_01(img_local_sub),
			# rescale_to_01(labelled)
			]]))
		plt.show()

	

	return labelled, local_min_coords



