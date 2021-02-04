from os import getcwd
from os.path import isfile
import numpy as np
import nibabel as nb
from scipy import ndimage

def contribution(s_index, experiments, tasks, study):
    
    cwd = getcwd()
    mask_folder = cwd + "/MaskenEtc/"
    try:
        os.mkdir(cwd + "/ALE/Contribution")
    except:
        pass
    
    s0 = list(range(len(s_index)))
    print(s0)
    
    template = nb.load(mask_folder + "Grey10.nii")
    template_data = template.get_fdata()
    template_shape = template_data.shape
    pad_tmp_shape = [value+30 for value in template_shape]

    exp_data = np.empty((len(s0), template_shape[0], template_shape[1], template_shape[2]))
    for i in s0:
        data = np.zeros(pad_tmp_shape)
        for ii in range(experiments.at[i, 'Peaks']):
            coords = experiments.XYZ[i].T[:,:3][ii]
            x_range = (coords[0],coords[0]+31)
            y_range = (coords[1],coords[1]+31)
            z_range = (coords[2],coords[2]+31)
            data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = \
            np.maximum(data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]],
                       experiments.at[i, 'Kernel'])
        exp_data[i,:,:,:] = (data[15:data.shape[0]-15,15:data.shape[1]-15, 15:data.shape[2]-15])

    for corr_method in ["TFCE", "FWE", "cFWE"]:
        txt = open(cwd + "/ALE/Contribution/" + study + "_" + corr_method + ".txt", "w+")
        txt.write("\nStarting with {}! \n".format(study))
        txt.write("\n{}: {} experiments; {} unique subjects (average of {:4.1f} per experiment) \n".format(study, len(s0), experiments.Subjects.sum(), experiments.Subjects.mean()))

        if isfile(cwd + "/ALE/Results/{}_{}05.nii".format(study, corr_method)):
            results = nb.load(cwd + "/ALE/Results/{}_{}05.nii".format(study, corr_method)).get_fdata()
            if results.any() > 0:
                labels, cluster_count = ndimage.label(results)
                ale = nb.load(cwd + "/ALE/Volumes/{}.nii".format(study))
                for label in np.unique(labels[labels > 0]):
                    clust_ind = np.vstack(np.where(labels == label))
                    clust_size = clust_ind.shape[1]
                    center = np.median(np.dot(template.affine, np.pad(clust_ind, ((0,1),(0,0)), constant_values=1)), axis=1)
                    if clust_ind[0].size > 5:
                        txt.write("\n\nCluster {}: {} voxel [Center: {}/{}/{}] \n".format(label, clust_size, int(center[0]), int(center[1]), int(center[2])))

                        ax = exp_data[:, clust_ind[0], clust_ind[1], clust_ind[2]]
                        axf = 1-np.prod(1-ax, axis=0)
                        axr = np.array([1-np.prod(1-np.delete(ax, i, axis=0), axis=0) for i in s0])
                        wig = np.array([np.sum(exp_data[i][tuple(clust_ind)]) for i in s0])
                        xsum = np.array([[wig[i], 100*wig[i]/clust_size, 100*(1-np.mean(np.divide(axr[i,:], axf))), np.max(100*(1-np.divide(axr[i,:], axf)))] for i in s0])
                        xsum[:,2] = xsum[:,2]/np.sum(xsum[:,2])*100

                        for i in s0:
                            if xsum[i, 2]>.1 or xsum[i, 3]>5:
                                stx = list(" " * (experiments.Author.str.len().max() + 2))
                                stx[0:len(experiments.Author[i])] = experiments.Author[i]
                                stx = "".join(stx)
                                txt.write("{}\t{:.3f}\t{:.3f}\t{:.2f}\t{:.2f}\t({})\n".format(stx,xsum[i,0],xsum[i,1],xsum[i,2],xsum[i,3],experiments.at[i, "Subjects"],))

                        txt.write("\n\n")

                        for i in range(tasks.shape[0]):
                            stx = list(" " * (tasks.Name.str.len().max()))
                            stx[0:len(tasks.Name[i])] = tasks.Name[i]
                            stx = "".join(stx)
                            mask = [s in tasks.ExpIndex[i] for s in s_index]
                            if mask.count(True) > 1:
                                xsum_tmp = np.sum(xsum[mask], axis=0)
                                txt.write("{}\t{:.3f}\t{:.3f}\t{:.2f}\t \n".format(stx,xsum_tmp[0],xsum_tmp[1], xsum_tmp[2]))
                            elif mask.count(True) == 1:
                                txt.write("{}\t{:.3f}\t{:.3f}\t{:.2f}\t \n".format(stx,xsum[mask][0,0],xsum[mask][0,1], xsum[mask][0,2]))
                            else:
                                pass

                txt.write("\nDone with {}!".format(corr_method))
                txt.close()
            else:
                txt.write("No significant clusters in {}!".format(corr_method))
                txt.close()
        else:
            txt.write("Could not find {} results for {}!".format(corr_method, study))
            txt.close()