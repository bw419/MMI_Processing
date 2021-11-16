from .imports import *
from .image_utils import *




def get_rot_angle(img, downscale_f=0.25, extra_precise=False, print_progress=True):
	''' 
	Attempts to find the correct angle to straighten the image, and returns it.
	First, attempts to find main trench's side's angle by using a Hough
	transform. 
	Optionally [if extra_precise is True], then performs optimisation in
	a bracket around this angle to maximise the sum of the squared gradient
	of the intensity profile projected onto x (i.e. aligned trenches.)
	
	Parameters:
	downscale_f (float) : Factor by which to downscale the image for speed. Default 0.25.
	extra_precise (bool) : Discussed above. Default False.
	print_progress (bool) : whether to print progress. Default True.

	'''

	ds_img = ski.transform.rescale(img, downscale_f)

	if print_progress: print("filtering")
	ds_filtered = ski.filters.difference_of_gaussians(ds_img, 5)

	if print_progress: print("thresholding")
	t = ski.filters.threshold_otsu(ds_filtered)

	ds_filtered[ds_filtered<t] = 0

	if print_progress: print("Alignment stage 1")

	hspace, angles, dists = ski.transform.hough_line(ds_filtered, theta=None)
	hspace, angles, dists = ski.transform.hough_line_peaks(hspace, angles, dists)

	angles *= 180/np.pi

	angle0 = (((angles[0] + 180)) % 180) - 90

	# If side trenches were detected as dominant lines:
	if angle0 > 45:
		angle0 = angle0 - 90

	dist = dists[0]

	if print_progress: print("angle, dist, img shape:", angle0, dist, img.shape)

	best_angle = angle0


	if extra_precise:
		if print_progress: print("Alignment stage 2")

		def proj_square_integral(angle):
			return -np.sum(np.square(np.gradient(ski.transform.rotate(ds_img, angle).sum(0))))

		out = scipy.optimize.minimize_scalar(proj_square_integral, method="bounded", bounds=[angle0-3, angle0+3], options={'xatol':1e-3})
		best_angle = out.x
		if print_progress: print(angle0, "--> more precise angle:", best_angle)

	return best_angle


def get_rot_transform(angle):

	def transform_img(img):
		return ski.transform.rotate(img, angle)

	return transform_img



def calc_and_return_rot_transform(img, downscale_f=0.25, extra_precise=False, print_progress=True):
	return get_rot_transform(get_rot_angle(img, downscale_f, extra_precise, print_progress))

