# %%
# %load_ext autoreload
# %autoreload 2
# %aimport -np, -plt, -ndi, -ski, -skimage, -pd, -pickle, -pathlib

from MMIData import MMIData, MMImageSeries, SegmentedMMI, Trench
from matplotlib import pyplot as plt
import numpy as np

# %%

Data = MMIData("C:/~/Documents/Proj_local/data", "binary_expt1")

if False:
	for FOV in range(1, Data.n_FOVs):
		for t in range(0, Data.n_timesteps):

			print(f"Timestep {t}, FOV {FOV}")
			
			segmented = Data.segment_single(t, FOV, True)

			# print("Got:", segmented)


			# # plt.imshow(np.hstack(segmented.load_masks()))
			# # plt.show()
			# trenches = segmented.load_masked_FL(0)
			# plt.imshow(trenches[0])
			# plt.show()

			# if t==1:
				# break
			# if t > 30:
				# break

		# break


# %%

blnk = Trench.get_blank_central_img()
base_colour = [np.array(x) for x in [[1,1,1], [0,0,1], [1,1,0], [1,0,0]]]
offsets = [(0,0), (0,0), (-1,-3), (-1,0)]


# %%
# Do a plot of the same trench in increasing time order.

for trench_n in range(1, 2):

	for FOV in range(3,4):

		fig, axs = plt.subplots(1, 2)
		axs = axs.flatten()
		# axs = [axs]


		j = 0 
		for ch in range(2, 4):

			plt.sca(axs[j])

			segmented = [SegmentedMMI.load(fp) if fp is not None else None for fp in Data.seg[:,FOV,"segmented_meta"][:45]]

			if segmented[0] is None:
				continue

			trench_ids = segmented[0].get_trench_ids()


			# break
			# print(Data.seg[:,FOV,"segmented_meta"])
			trench_id = trench_ids[3]

			title_str = f"{Data.imgs.channel_names[ch]} - FOV {FOV} - trench {trench_id}"

			ids = [seg.get_trench_idx_by_id(trench_id)
				if (seg is not None) else None
				for i, seg in enumerate(segmented) if i%5==0]


			trench1s = [seg.load_masked_channel(ch, offset=offsets[ch], dilate=3)[ids[i//5]].T 
				if (seg is not None) and (ids[i//5] is not None) else blnk 
				for i, seg in enumerate(segmented) if i%5==0]

			print("Trenches:", len(trench1s))


			plt.title(title_str)
			img = np.vstack(trench1s)
			img /= np.max(img)
			img_c = np.zeros(img.shape + (3, ))
			img_c = img[:,:,np.newaxis] * base_colour[ch]

			plt.imshow(img_c)

			j += 1

		plt.show()


# %%

for FOV in range(1,2):

	for ch in range(2, 4):

		print("loading...")
		segmented = [SegmentedMMI.load(fp) if fp is not None else None for i, fp in enumerate(Data.seg[:,FOV,"segmented_meta"][:30]) if (i%5)==0][3]
		trench_masks = segmented.load_masks(dilate=5)#[s.load_masks() if (s is not None and len(s.trenches) > 3) else blnk for s in segmented]
		trench_masks_ch = segmented.load_masked_channel(ch, offset=offsets[ch], dilate=5)

		print("loaded")
		# trench_masks_ch = [seg.load_masked_channel(ch, offset=offsets[ch])[trench_n].T 
		# 						if (seg is not None and len(seg.trenches) > 3) else blnk 
		# 						for k, seg in enumerate(segmented)]

		for trench_n in range(1,2):



			trench_mask = trench_masks[trench_n].T
			trench_mask_ch = trench_masks_ch[trench_n].T

			title_str = f"{Data.imgs.channel_names[ch]} - FOV {FOV} - trench {trench_n}"
			print(title_str)

			A = []
			B = []
			C = []
			D = []

			pos = []

			for cell_idx in range(trench_mask.max()+1):

				cell_mask = trench_mask==cell_idx
				cell = trench_mask_ch[cell_mask]


				mean_intensity = cell.mean()
				max_intensity = cell.max()
				min_intensity = cell.min()
				std = cell.std()
				mean_x = np.mean(np.argwhere(cell_mask), axis=0)[1]


				A.append(mean_intensity)
				B.append(max_intensity)
				C.append(min_intensity)
				D.append(std)
				pos.append(mean_x)



			# for k, p in enumerate(pos):
			# 	plt.plot([p, p], [B[k], C[k]], "k--", lw=1)
			from matplotlib.ticker import FormatStrFormatter
			f1, f2 = plt.figure(), plt.figure()


			plt.errorbar(pos[1:], A[1:], D[1:], linestyle='None', marker='x')
			plt.xlim(0, trench_mask_ch.shape[1])
			plt.plot([10, trench_mask_ch.shape[1]-10], [A[0], A[0]], "k--")
			plt.show()

			img = trench_mask_ch
			img /= np.max(img)
			img_c = np.zeros(img.shape + (3, ))
			img_c = img[:,:,np.newaxis] * base_colour[ch]

			plt.imshow(img_c)
			plt.show()
			plt.imshow(trench_mask)
			plt.show()

	# labelled = ski.segmentation.expand_labels(labelled, 5) useful command










# %%
import copy

dilation = None

FOV = 1
# for FOV in range(1,2):
#%%
	
	print("loading all SegmentedMMI objects...")
	segmented = [SegmentedMMI.load(fp) if fp is not None else None for i, fp in enumerate(Data.seg[:,FOV,"segmented_meta"]) if (i%5)==0]

	ids0 = segmented[0].get_trench_ids()
	n_trenches = len(ids0)

	D = []


	for t in range(0, 29):

		if segmented[t] is None:
			continue

		print(f"Time {t}...")
		trench_masks = segmented[t].load_masks(dilate=dilation)
		#[s.load_masks() if (s is not None and len(s.trenches) > 3) else blnk for s in segmented]

		if len(trench_masks) != n_trenches:
			print(f"Not the right number of trenches. {len(trench_masks)}/{n_trenches}.")
			continue

		ch_range = [2, 3]

		empty1 = {ch : {"mean":[], "std":[]} for ch in ch_range}
		empty = {"cell_idx":[], "pos":[], "strain":[]}
		empty.update(empty1)
		trench_data_map = [copy.deepcopy(empty) for tr in range(len(trench_masks))]

		bg_means, bg_stds = segmented[t].get_FL_bg_vals()
		trench_bg_vals = {ch+1: {"mean" : bg_mean, "std" : bg_std} for ch, (bg_mean, bg_std) in enumerate(zip(bg_means, bg_stds))}
		# print(trench_bg_vals)

		trench_data_map = trench_data_map#[0:4]
		trench_masks = trench_masks#[0:4]

		for ch in ch_range:

			trench_masks_ch = segmented[t].load_masked_channel(ch,  dilate=dilation, offset=offsets[ch])

			trench_masks_ch = trench_masks_ch#[0:4]

			# trench_masks_ch = [seg.load_masked_channel(ch, offset=offsets[ch])[trench_n].T 
			# 						if (seg is not None and len(seg.trenches) > 3) else blnk 
			# 						for k, seg in enumerate(segmented)]

			for trench_n in range(0,len(trench_masks)):

				trench_mask = trench_masks[trench_n].T
				trench_mask_ch = trench_masks_ch[trench_n].T

				img = trench_mask_ch.copy()

				# plt.imshow(trench_mask)
				# plt.show()

				# img[trench_mask!=1] = 0
				# plt.imshow(img)
				# plt.show()
				# img[trench_mask==1] = 1
				# plt.imshow(img)
				# plt.show()

				for cell_idx in range(1, trench_mask.max()+1):
					# print(trench_n, cell_idx)
					cell_mask = trench_mask==cell_idx

					if cell_mask.sum() == 0 :
						# print(f"continuing for idx {cell_idx}, trench {trench_n}, channel {ch}")
						continue

					cell = trench_mask_ch[cell_mask]

					# if trench_n == 0 and cell_idx == 1:
					# 	plt.hist(cell.flatten())
					# 	plt.show()

					if ch == ch_range[0]:
						trench_data_map[trench_n]["cell_idx"].append(cell_idx)
						trench_data_map[trench_n]["pos"].append(np.mean(np.argwhere(cell_mask), axis=0)[1])

					trench_data_map[trench_n][ch]["mean"].append(cell.mean())
					trench_data_map[trench_n][ch]["std"].append(cell.std())


				sort_idxs = np.argsort(trench_data_map[trench_n]["pos"])

				# print(f"------ch{ch}-1------")
				# print(trench_data_map[trench_n]["pos"])
				# print(trench_data_map[trench_n][ch]["mean"])

				for key, val in trench_data_map[trench_n][ch].items():
					trench_data_map[trench_n][ch][key] = list(np.array(val)[sort_idxs])

				# print(f"------ch{ch}-2------")
				# print(trench_data_map[trench_n]["pos"])
				# print(trench_data_map[trench_n][ch]["mean"])


		for trench_n in range(0,len(trench_masks)):

			sort_idxs = np.argsort(trench_data_map[trench_n]["pos"])

			for key, val in trench_data_map[trench_n].items():
				if key in ["cell_idx", "pos"]:
					trench_data_map[trench_n][key] = list(np.array(val)[sort_idxs])

		D.append((t, trench_data_map))



# %%

	strains = {"A" : 0, "B" : 1, "C" : 2, "B or C": 3, "None" : -1}
	strain_colours = {"A" : "b", "B" : "g", "C": "c", "B or C": "m", "None" : "k"}

	for t, trench_data_map in D:

		for tdata in trench_data_map:

			if len(tdata["cell_idx"]) == 0:
				continue

			y = 2
			r = 3
			trench_thresholds = {ch: val["mean"] + val["std"] for ch, val in trench_bg_vals.items() }

			tdata["strain"] = -np.ones_like(tdata["cell_idx"])



			for i, pos in enumerate(tdata["pos"]):
				r_mean = tdata[r]["mean"][i]
				y_mean = tdata[y]["mean"][i]

				if r_mean > trench_thresholds[r]:
					
					if y_mean > trench_thresholds[y]:
						# tdata["strain"][i] = strains["B"]
						if (y_mean-trench_bg_vals[y]["mean"]) < 0.5*(r_mean-trench_bg_vals[r]["mean"]):
							tdata["strain"][i] = strains["B"]
						else:
							tdata["strain"][i] = strains["None"]
					else:
						tdata["strain"][i] = strains["B or C"]

				elif y_mean > trench_thresholds[y]:
					tdata["strain"][i] = strains["A"]

				else:
					tdata["strain"][i] = strains["None"]



			new_strain_arr = tdata["strain"].copy()
			for i, pos in enumerate(tdata["pos"][:-1]):
				s = tdata["strain"][i]
				s1 = tdata["strain"][i+1]


				if s != s1 and (strains["None"] not in tdata["strain"][i:i+2])\
					and not ((strains["B or C"] == s and strains["B"] == s1) or\
					 		 (strains["B or C"] == s1 and strains["B"] == s)):

					new_strain_arr[i] = strains["None"]
					new_strain_arr[i+1] = strains["None"]

			if len(new_strain_arr) > 0:
				new_strain_arr[0] = strains["None"]
				new_strain_arr[-1] = strains["None"]
	
			tdata["strain"] = new_strain_arr


# %%
	
	t_d_map_1 = D[0][1]

	# Disregard a trench if it contains strain C in the first timestep.
	contains_strain_c = np.zeros(len(t_d_map_1), dtype=bool)

	for i, tdata in enumerate(t_d_map_1):
		if strains["B or C"] in tdata["strain"]:
			contains_strain_c[i] = True

	for t, trench_data_map in D:

		for i, tdata in enumerate(trench_data_map):
			if not contains_strain_c[i]:
				for j, s in enumerate(tdata["strain"]):
					if s == strains["B or C"]:
						tdata["strain"][j] = strains["B"]


	print("Trenches containing strain C:", contains_strain_c)

# %%
	
	D1 = []
	for t, trench_data_map in D:
		sA_y_intensities = []
		sB_y_intensities = []
		sB_r_intensities = []

		for i, tdata in enumerate(trench_data_map):
			if not contains_strain_c[i]:
				for j, s in enumerate(tdata["strain"]):
					if s == strains["A"]:
						sA_y_intensities.append(tdata[y]["mean"][j])# - trench_bg_vals[y]["mean"])
					elif s == strains["B"]:
						sB_y_intensities.append(tdata[y]["mean"][j])# - trench_bg_vals[y]["mean"])
						sB_r_intensities.append(tdata[r]["mean"][j])# - trench_bg_vals[r]["mean"])

		D1.append((sA_y_intensities, sB_y_intensities, sB_r_intensities))


	print(D1[0][0])

# %%

	# Scatter of all samples over time. 

	ts = np.array([x[0] for x in D])*10*5/60

	D2 = [[np.mean(x[i]) if len(x[i]) > 0 else np.nan for x in D1 ] for i in range(3)]
	D3 = [[np.std(x[i]) if len(x[i]) > 0 else np.nan for x in D1 ] for i in range(3)]

	t1s = []
	D4 = []
	for k in range(3):
		D4.append([])
		t1s.append([])
		for t, x in zip(ts, D1):
			for y in x[k]:
				t1s[k].append(t)
				D4[k].append(y)

		t1s[k] = np.array(t1s[k])

	y, r = 2, 3

	plt.plot(t1s[0]-0.03, D4[0], "yx")
	plt.plot(t1s[2], D4[2], "r.")
	# eb=plt.errorbar(ts+0.03, D2[1], D3[1], fmt="yo", linestyle="None", label="YFP intensity for Strain B")

	plt.plot([ts[0], ts[-1]], [trench_bg_vals[y]["mean"]]*2, "y--", label="YFP background")
	# plt.plot([ts[0], ts[-1]], [trench_bg_vals[y]["mean"]+trench_bg_vals[y]["std"]]*2, "y--", label="YFP background")
	# plt.plot([ts[0], ts[-1]], [trench_bg_vals[y]["mean"]]*2, "y--", label="YFP background")
	plt.plot([ts[0], ts[-1]], [trench_bg_vals[r]["mean"]]*2, "r--", label="RFP background")
	plt.semilogy()

	plt.plot([80./60.], [trench_bg_vals[y]["mean"]], "kD")
	plt.plot([80./60.]*2, [trench_bg_vals[y]["mean"], np.max(D2[2])], "k--", zorder=-5)

	
	# plt.twinx()


	plt.plot(t1s[1]+0.03, D4[1], "y.")

	# plt.ylim(0, 400)
	plt.show()

# %%

	maxes = [np.max([np.mean(x[i]) for x in D1 if len(x[i]) > 0]) for i in range(3)]

	# plt.plot(ts, D2[0], "yx", label="YFP intensity for Strain A")
	# plt.plot(ts, D2[2], "ro", label="RFP intensity for Strain B")
	# plt.plot([1], [np.nan], "yo", label="YFP intensity for Strain B")




	eb1=plt.errorbar(ts-0.03, D2[0], D3[0], fmt="yx-",  lw=1, label="YFP intensity for SB6")
	eb2=plt.errorbar(ts, D2[2], D3[2], fmt="ro-", lw=1, label="RFP intensity for SB111")
	# eb3=plt.errorbar([1], [np.nan], [np.nan], fmt="yo", linestyle=":", label="YFP intensity for Strain B")

	plt.plot([ts[0], ts[-1]], [trench_bg_vals[y]["mean"]]*2, "y--", label="YFP background")
	plt.plot([ts[0], ts[-1]], [trench_bg_vals[r]["mean"]]*2, "r--", label="RFP background")
	plt.semilogy()

	plt.plot([80./60.], [trench_bg_vals[y]["mean"]*.9], "kD")
	plt.plot([80./60.]*2, [trench_bg_vals[y]["mean"]*.9, np.max(D2[2])], "k--", zorder=-5)


	RHS = 15.0
	TOP = 2180
	BOT = TOP*0.9

	plt.plot([RHS-34.4/60,RHS], [TOP]*2, "r", lw=10)
	plt.plot([RHS-4.1/60,RHS], [BOT]*2, "y", lw=10)
	plt.plot([RHS-80.0/60,RHS], [TOP]*2, "r", lw=10,alpha=0.5)
	plt.plot([RHS-18.5/60,RHS], [BOT]*2, "y", lw=10,alpha=0.5)

	eb3=plt.errorbar(ts+0.03, D2[1], D3[1], fmt="yo-", lw=1, label="YFP intensity for SB111")
	eb[-1][0].set_linestyle(":")
	# eb[0].set_linewidth("1")

	# eb[0].set_c("k")
	plt.plot([np.nan], [np.nan], "k", lw=10, alpha=0.6, label="50% maturation time")
	plt.plot([np.nan], [np.nan], "k", lw=10, alpha=0.3, label="90% maturation time")


	plt.ylabel("log intensity value (a.u.)")
	plt.xlabel("Time into experiment (hrs)")

	plt.legend()




	plt.savefig("fig1.svg")

	plt.show()



# %%
# %%
# %%


	for t, trench_data_map in D:

		for i, trench_data in enumerate(trench_data_map):

			if len(trench_data["cell_idx"]) == 0:
				continue


			colours = {2:"y", 3:"r"}
			for ch in ch_range:

		# labelled = ski.segmentation.expand_labels(labelled, 5) useful command
				plt.title(f"Strain intensities in trench {i}, t={t}")
				plt.errorbar(trench_data["pos"], trench_data[ch]["mean"], trench_data[ch]["std"], fmt=colours[ch], linestyle="None", marker='x')
				# plt.xlim(0, trench_mask_ch.shape[1])

				bg_mean = trench_bg_vals[ch]["mean"]
				bg_std = trench_bg_vals[ch]["std"]
				min_to_max = [min(trench_data["pos"]), max(trench_data["pos"])]
				plt.plot(min_to_max, [bg_mean]*2, colours[ch]+"--")
				plt.plot(min_to_max, [bg_mean+bg_std]*2, colours[ch], lw=1)
				plt.plot(min_to_max, [bg_mean-bg_std]*2, colours[ch], lw=1)

				for s in strains:
					for k, strain_id in enumerate(trench_data["strain"]):
						if strains[s] == strain_id:
							plt.plot(trench_data["pos"][k], [0], strain_colours[s] + "o")

			plt.show()




			# for ch in trench_data:
			# 	img = trench_mask_ch
			# 	img /= np.max(img)
			# 	img_c = np.zeros(img.shape + (3, ))
			# 	img_c = img[:,:,np.newaxis] * base_colour[ch]
			# plt.imshow(img_c)
			# plt.show()
			# plt.imshow(trench_mask)
			# plt.show()

