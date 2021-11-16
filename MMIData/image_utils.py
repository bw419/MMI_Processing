from .global_imports import *

# unused
def cutoff_n_sds_above_mean(img, n=1):
	m = img.mean()
	s = img.std()
	img[img > m + n*s] = m + n*s
	return img

# uneccessary, just use inner function
def get_local_maxima(img, show_img=False, min_distance=1):

	local_maxima_coords = ski.feature.peak_local_max(img, min_distance)
	# local_maxima = np.zeros_like(img, dtype=bool)
	# local_maxima[tuple(local_maxima_coords.T)] = True

	if show_img:
		plt.imshow(img, cmap=plt.cm.gray)
		plt.plot(local_maxima_coords[:,1], local_maxima_coords[:,0], 'r.')
		plt.show()

	return local_maxima_coords


def rescale_to_01_inplace(img):
	img -= img.min()
	img /= img.max()


def rescale_to_01(img):
	return (img - img.min())/(img.max()-img.min())