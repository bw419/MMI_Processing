from .imports import *
from .data_utils import MA_zeropad


def TBLR_to_slice(TBLR):
	return np.s_[TBLR[0]:TBLR[1], TBLR[2]:TBLR[3]]

def offset_TBLR_slice(TBLR, offset):
	TBLR1 = TBLR[0] + offset[0], TBLR[1] + offset[0], TBLR[2] + offset[1], TBLR[3] + offset[1]
	return TBLR_to_slice(TBLR1)

class Trench():

	inner_trench_width = 24

	@classmethod
	def get_blank_central_img(cls):
		return np.zeros((cls.inner_trench_width, 620))

	def __init__(self, rotated_PC_img, TBLR):

		self.TBLR = TBLR
		self.slice = TBLR_to_slice(TBLR)

		# self.T, self.B, self.L, self.R = TBLR

		local_maxes = scipy.signal.argrelextrema(rotated_PC_img[self.slice].sum(0), np.less)[0]
		if len(local_maxes) == 0:
			plt.imshow(rotated_PC_img)
			plt.imshow(rotated_PC_img[self.slice].T)
			plt.imshow(rotated_PC_img[self.slice].T)
			plt.show()
			plt.plot(rotated_PC_img[self.slice].sum(0))
			plt.show()


		L1, R1 = local_maxes[0], local_maxes[-1]
		half_w = Trench.inner_trench_width*0.5
		delta_from_centre = np.array([-half_w, half_w])
		L1, R1 = np.round(0.5*(L1 + R1) + delta_from_centre).astype(int)

		self.central_TBLR = self.TBLR[0], self.TBLR[1], L1+self.TBLR[2], R1+self.TBLR[2]

		self.central_slice = TBLR_to_slice(self.central_TBLR)




class TrenchSignalDetector():

	this_path = Path(__file__).parent.resolve()
	
	default_gap_signal_path = this_path / "signal_to_find.pickle"
	sample_real_signal_path = this_path / "sample_real_data.pickle"

	# default_trench_width = 42 # does not currently affect anything
	# default_trench_innder_width = 42


	def __init__(self, gap_signal_fpath=None, gap_signal=None, real_data_signal_fpath=None, real_data_signal=None, trench_width=None):

		# if trench_width is not None:
		# 	self.trench_width = trench_width
		# else:
		# 	self.trench_width = TrenchSignalDetector.default_trench_width


		if gap_signal_fpath is not None:
			with open(signal_fpath, "rb") as f:
				self.gap_sig_to_find = pickle.load(f)
			return 

		if gap_signal is not None:
			self.gap_sig_to_find = gap_signal.copy()
			return


		if real_data_signal_fpath is not None:
			with open(signal_fpath, "rb") as f:
				x = pickle.load(f)
			self.gap_sig_to_find = self._create_clean_signal_to_find(x)
			return 

		if real_data_signal is not None:
			self.gap_sig_to_find = self._create_clean_signal_to_find(real_data_signal)
			return


		else:
			with open(TrenchSignalDetector.default_gap_signal_path, "rb") as f:
				self.gap_sig_to_find = pickle.load(f)
			return 


	def extract_trenches(self, rotated_PC_img, show_img=False):


		LRs, int_LRs = self._get_trench_LRs(
			rotated_PC_img, show_img
		)

		trenches = []

		for i, (L, R) in enumerate(int_LRs):

			T, B = self._get_trench_TB_crude(
				rotated_PC_img, L, R, show_img
			)

			trenches.append(Trench(rotated_PC_img, [T, B, L, R]))

		return trenches



	def _get_trench_LRs(self, rotated_PC_img, show_img=False):
		''' 
		Apply a Weiner(?) filter to the rotation-corrected image to find
		trench gap signals & extract trench edges
		'''  


		x_proj = rotated_PC_img.sum(0)
		ddx = np.gradient(x_proj) #OFFSET OF 0.5


		convolved = np.convolve(ddx, self.gap_sig_to_find[::-1], 'same')

		# TODO: PROPER PARAMS
		peaks = scipy.signal.find_peaks(convolved, height=30, distance=100)[0] 

		# TODO: PROPER PARAMS
		LEN_GAP = len(self.gap_sig_to_find)
		n_gaps = len(peaks)

		if n_gaps == 0:
			raise Exception("Couldn't detect any peaks...")

		gap_plus_trench_lens = peaks[1:] - peaks[:-1]

		mean = np.mean(gap_plus_trench_lens)
		std = np.std(gap_plus_trench_lens)

		if mean < 148 or mean > 162 or std > 2:
			raise Exception("Inconsistent peaks...") 

		gap_plus_trench_len = round(mean*2)/2 
		LEN_TRENCH = gap_plus_trench_len - LEN_GAP


		# print(n_gaps, gap_plus_trench_len, LEN_GAP)

		# 0.5 offset because of finite difference differentiation
		LHSs = [peaks[0] - LEN_TRENCH - LEN_GAP/2 + 0.5] + list(peaks + LEN_GAP/2 + 0.5)
		RHSs = list(peaks - LEN_GAP/2 + 0.5) + [peaks[-1] + LEN_TRENCH + LEN_GAP/2 + 0.5]


		if LHSs[0] < 0:
			LHSs = LHSs[1:]
			RHSs = RHSs[1:]
		if RHSs[-1] >= x_proj.size-1:
			LHSs = LHSs[:-1]
			RHSs = RHSs[:-1]


		trench_full_LRs = list(zip(LHSs, RHSs))
		# for i, (L, R) in enumerate(trench_full_LRs):
			# trench_full_LRs[i] = 0.5*(R + L) + np.array([-self.trench_width//2, self.trench_width//2])

		int_trench_full_LRs = [[int(np.round(L)), int(np.round(R))] for L, R in trench_full_LRs]

		if show_img:
			plt.plot(x_proj-np.mean(x_proj))
			plt.plot(convolved, zorder=1)
			plt.plot(peaks, [0]*len(peaks), "rx", ms=10)

			for lhs in LHSs:
				plt.plot([lhs]*2, [-40, 40], "g--")
			for rhs in RHSs:
				plt.plot([rhs]*2, [-40, 40], "g--")

			plt.show()

		return trench_full_LRs, int_trench_full_LRs



	def _get_trench_TB_crude(self, rotated_PC_img, trench_L, trench_R, show_img=False):

		trench_img = rotated_PC_img[:,trench_L:trench_R]

		projy = trench_img.sum(1)

		s_projy = MA_zeropad(projy, 5)
		s_ddy = np.gradient(s_projy)

		AC_projy = s_projy - s_projy.mean()		
		AC_projy > AC_projy.max()*0.5



		T = np.argmax(s_projy) + 30

		B_range = [T + 600, T + 680]

		B = B_range[0] + np.argmin(s_ddy[B_range[0]:B_range[1]]) - 30
		T = B - 620

		if show_img:
			plt.imshow(trench_img.T)
			plt.plot(s_projy, "r")
			plt.plot((s_ddy-s_ddy.min())*5, "g")
			plt.plot([T, B], [25]*2, "m.")
			plt.show()

			plt.imshow(trench_img.T[:,T:B])

			plt.show()

		return T, B





	# some real data in "good_signal_real_data.pickle"
	def _create_clean_signal_to_find(real_data):
		''' 
		Fit a parameterised function to some real data to obtain a 'clean' signal to search for
		'''

		def gaussian(x, mu, sig):
		    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

		def model_fn(params):
			m, sf, mu, sd = params
			return lambda x: m*x - sf*gaussian(x, -mu, sd) +  sf*gaussian(x, mu, sd)

		def objective(params):
			x = np.linspace(-1, 1, 100)
			return np.sum(np.square(model_fn(params)(x) - SIG_TO_FIND))

		out = scipy.optimize.minimize(objective, [-0.07, 0.5, 0.89, 0.04])
		print(out)

		y = model_fn(out.x)(x)

		plt.plot(SIG_TO_FIND)
		plt.plot(np.arange(len(y)), y)
		plt.show()

		return y


	def save_signal_as_default(self):

		with open(self.default_gap_signal_path) as f:
			pickle.dump(self.gap_sig_to_find, f)
