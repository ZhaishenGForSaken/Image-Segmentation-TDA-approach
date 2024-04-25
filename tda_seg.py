import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import gudhi as gd
from scipy.ndimage import gaussian_filter
import networkx as nx
from sklearn.neighbors._kde import KernelDensity
import time
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage import exposure
from skimage.util import img_as_float
from skimage.color import rgb2gray, gray2rgb


class Tree(object):
    def __init__(self, graph=None):
        self.graph = graph
        self.parent = None
        self.children = None
        if not not self.graph:
            self.pixels = graph.nodes
        else:
            self.pixels = None
        self.depth = 0
        self.state = 0
        self.key = 0
        self.count = 1
        self.out_pixels = None

    def get_root(self, count='no'):
        if self.depth == 0:
            return self
        else:
            node = self;
            if count != 'yes':
                for i in range(self.depth):
                    node = node.parent
            else:
                for i in range(self.depth):
                    node = node.parent
                    node.count = node.count + 1
            return node

    def get_leaves(self):
        if not self.children:
            yield self

        else:
            for child in self.children:
                for leaf in child.get_leaves():
                    yield leaf

    def get_depth(self):
        depth = 0;
        for leaf in self.get_leaves():
            if leaf.depth > depth:
                depth = leaf.depth
        return depth

    def expand(self, list_edges_to_be_removed, size=10, proba=0.05):
        out_pixels = []
        H = self.graph
        for leaf in self.get_leaves():

            if leaf.state == 0:

                subH = nx.Graph(H.subgraph(leaf.pixels));

                if set(subH.edges()) & set(list_edges_to_be_removed):

                    subH.remove_edges_from(list_edges_to_be_removed)
                    cc = [list(a) for a in nx.connected_components(subH)]

                    if len(cc) > 1:

                        if size > 0:

                            nodes_toberemoved = [-1] * len(cc)
                            loss = 0;
                            n_nodes_tbr = 0
                            max_size_c = size
                            for j, c in enumerate(cc):
                                size_c = len(c)
                                if size_c <= size:  # small cc
                                    nodes_toberemoved[n_nodes_tbr] = j
                                    loss += size_c
                                    n_nodes_tbr += 1
                                if size_c > max_size_c:  # big cc
                                    max_size_c = size_c
                            nodes_toberemoved = nodes_toberemoved[:n_nodes_tbr]
                            cpt_not_small = len(cc) - len(nodes_toberemoved)
                            bool_expand = loss / len(leaf.pixels) <= proba or self.get_depth() == 0
                            bool_pause = (max_size_c / len(
                                leaf.pixels) >= 1. - proba and not self.get_depth() == 0 and max_size_c / len(
                                self.get_root().pixels) < 0.3)
                        else:
                            nodes_toberemoved = []
                            bool_expand = True
                            bool_pause = False
                            cpt_not_small = 1000

                        if bool_expand and cpt_not_small > 0 and not bool_pause:  # expand
                            if len(nodes_toberemoved) > 0:
                                out_pixels = out_pixels + [cc[i] for i in nodes_toberemoved]
                            cc = [cc[i] for i in range(len(cc)) if i not in nodes_toberemoved]
                            leaf.children = [Tree() for c in cc]
                            for i, child in enumerate(leaf.children):
                                child.pixels = cc[i]
                                child.parent = leaf
                                child.depth = leaf.depth + 1
                                child.key = child.get_root(count='yes').count - 1
                        elif bool_pause:
                            leaf.state = 0

                        else:
                            leaf.state = 1
        if not self.get_root().out_pixels:
            self.get_root().out_pixels = [out_pixels]
        else:
            self.get_root().out_pixels += [out_pixels]

    def as_str(self, level=0):
        ret = "\t" * level + repr(len(self.pixels)) + "\n"
        if not not self.children:
            for child in self.children:
                ret += child.as_str(level + 1)
        return ret


def segment_with(I, seg_method):
    # PARAMETERS

    # Felzenswalb
    scale_ = 400;
    sigma_ = .5;
    min_size_ = 500;

    # SLIC
    n_segments_ = 100;  # 15
    compactness_ = 10;
    sigma_ = 1

    # Quickshift
    kernel_size_ = 20
    max_dist_ = 45
    ratio_ = 0.5

    # Watershed
    markers_ = 10
    compactness_ = 0.001

    # SEGMENTATION METHODS

    if seg_method == 'Otsu Thresholding':
        I = rgb2gray(I);
        thd = threshold_otsu(I);
        Ib = I < thd;
        plt.imshow(Ib, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

        hist, bins_center = exposure.histogram(I)
        plt.plot(bins_center, hist, lw=2)
        plt.axvline(thd, color='r', ls='--')
        plt.show()
        return Ib

    # FELZENSWALB'S METHOD
    if seg_method == 'Felzenswalb':

        plt.axis('off')
        plt.title('Felzenswald\'s method')
        segments_fz = felzenszwalb(I, scale_, sigma_, min_size_);
        plt.imshow(mark_boundaries(I, segments_fz))
        plt.show()
        J = I;
        for l in range(np.max(segments_fz) + 1):
            J[segments_fz == l] = np.mean(I[segments_fz == l], axis=0);
        if len(I.shape) == 2:
            plt.imshow(J, cmap='gray', interpolation='nearest');
        else:
            plt.imshow(J);
        plt.axis('off')
        plt.show()
        return segments_fz

    # SLIC'S METHOD
    if seg_method == 'SLIC':

        plt.axis('off')
        plt.title('SLIC\'s method')
        segments_slic = slic(I, n_segments=n_segments_, compactness=compactness_, sigma=sigma_)
        plt.imshow(mark_boundaries(I, segments_slic))
        plt.show()
        J = I;
        for l in range(np.max(segments_slic) + 1):
            J[segments_slic == l] = np.mean(I[segments_slic == l], axis=0);
        if len(I.shape) == 2:
            plt.imshow(J, cmap='gray', interpolation='nearest');
        else:
            plt.imshow(J);
        plt.axis('off')
        plt.show()
        return segments_slic

    # QUICKSHIFT'S METHOD
    if seg_method == 'Quickshift':
        plt.axis('off')
        plt.title('Quickshift\'s method')
        segments_quick = quickshift(I, kernel_size=kernel_size_, max_dist=max_dist_, ratio=ratio_)
        plt.imshow(mark_boundaries(I, segments_quick))
        plt.show()
        return segments_quick

    #  WATERSHED'S METHOD
    if seg_method == 'Watershed':
        plt.axis('off')
        plt.title('Watershed\'s method')
        gradient = sobel(rgb2gray(I));
        segments_watershed = watershed(gradient, markers=markers_, compactness=compactness_);
        plt.imshow(mark_boundaries(I, segments_watershed));
        plt.show()

        return segments_watershed


def pd_segmentation(mode, n_superpixels_, img, rv_epsilon=30, gaussian_sigma=0.5, list_events_=[350], min_pixel=10,
                    entropy_thresh=0.05, density_excl=0.05, plot_pd=False):
    height, width = img.shape[:2]
    height_0 = height
    width_0 = width
    if len(img.shape) > 2:
        n_col_chan = 3
    else:
        n_col_chan = 1

    n_pixels = np.prod(img.shape[:2])
    list_squares = [4, 9, 16, 25, 36, 49, 64]
    list_pos = [(1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)]
    list_steps = [2, 3, 4, 5, 6, 7, 8]

    if mode == 'standard':
        gaussian_sigma = 0.5;
        min_pixel = 15;
        entropy_thresh = .15;
        density_excl = 0.;
        plot_pd_ = False;
        list_squares = [4, 9, 16, 25, 36, 49, 64]  # [25, 36, 49, 64];
        list_pos = [(1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)]  # [(2,2), (3,2), (3,3), (4,3)];
        list_steps = [2, 3, 4, 5, 6, 7, 8]  # [5, 6, 7, 8];

    step_idx = np.argmin([np.abs(n_pixels / s - n_superpixels_) for s in list_squares])
    step = list_steps[step_idx];

    step_up_j = list_pos[step_idx][0]
    step_down_j = list_pos[step_idx][1]

    step_up_i = list_pos[step_idx][1]
    step_down_i = list_pos[step_idx][0]

    dh = int(np.ceil(height / step)) * step - height
    dw = int(np.ceil(width / step)) * step - width

    if dh > 0:
        dhI = img[-dh:, :]
        img = np.concatenate((img, dhI[::-1, :]), axis=0)
    if dw > 0:
        dwI = img[:, -dw:]
        img = np.concatenate((img, dwI[:, ::-1]), axis=1)

    grid_y, grid_x = np.mgrid[:img.shape[0], :img.shape[1]]
    means_y = grid_y[list_pos[step_idx][0]::step, list_pos[step_idx][1]::step]
    means_x = grid_x[list_pos[step_idx][0]::step, list_pos[step_idx][1]::step]

    if gaussian_sigma > 0:
        img_blurred = gaussian_filter(img, sigma=gaussian_sigma * np.floor(0.5 * step) / 4)
    else:
        img_blurred = img

    # from image to cloud point data
    point_cloud_data = np.dstack((means_y, means_x, img_blurred[means_y, means_x] * 255)).reshape((-1, n_col_chan + 2))

    num_points = point_cloud_data.shape[0]

    if density_excl > 0:
        kde = KernelDensity(kernel='gaussian', bandwidth=20).fit(point_cloud_data)
        pcd_density = kde.score_samples(point_cloud_data)
        sorted_density = sorted(pcd_density, reverse=True)
        n_excl = int(num_points * density_excl)
        thresh_density = sorted_density[-n_excl:][0]
        excl_pcd = point_cloud_data[pcd_density <= thresh_density, :2]
        point_cloud_data = point_cloud_data[pcd_density > thresh_density, :]
        num_points = point_cloud_data.shape[0]
    else:
        excl_pcd = np.zeros((0, 2))

    if mode == 'standard':
        ratio = np.prod(height_0 * width_0) / num_points
        rv_epsilon = np.ceil(0.5 * ratio + 10)
    # print('RV Eplison: ', rv_epsilon)
    Rips_complex_sample = gd.RipsComplex(points=point_cloud_data, max_edge_length=rv_epsilon)
    Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=1)
    Rips_Pd = Rips_simplex_tree_sample.persistence()

    if plot_pd == True:
        diag_Rips_0 = Rips_simplex_tree_sample.persistence_intervals_in_dimension(0)
        print('lamost plot')
        plt = gd.plot_persistence_diagram([(0, interval) for interval in diag_Rips_0], max_plots=0, alpha=0.1,
                                          legend=True)
        plt.show()

    persistence_pairs = Rips_simplex_tree_sample.persistence_pairs()

    betti_0 = Rips_simplex_tree_sample.betti_numbers()[0]

    key_edges_0 = [tuple(pair[1]) for pair in Rips_simplex_tree_sample.persistence_pairs() if len(pair[0]) == 1][
                  :-betti_0];

    G = nx.Graph()
    G.add_nodes_from(range(num_points))
    G.add_edges_from(key_edges_0)

    if mode == 'standard':

        if min_pixel * step * step <= 200 and ratio < 10:
            list_events_ = [500, 1000]
        if min_pixel * step * step > 200 and ratio < 10:
            list_events_ = [200, 500, 1000]

        if min_pixel * step * step <= 100 and 10 <= ratio < 18:
            list_events_ = [600]
        if 100 < min_pixel * step * step <= 200 and 10 <= ratio < 18:
            list_events_ = [200, 600]
        if min_pixel * step * step > 200 and 10 <= ratio < 18:
            list_events_ = [125, 300, 700]

        if min_pixel * step * step <= 100 and ratio > 18:
            list_events_ = [500]
        if 100 <= min_pixel * step * step < 200 and ratio > 18:
            list_events_ = [200, 500]
        if 200 <= min_pixel * step * step < 300 and ratio > 18:
            list_events_ = [150, 500]
        if min_pixel * step * step >= 300 and ratio > 18:
            list_events_ = [75, 275, 500]

        c0 = (list_events_[0] - betti_0) * (betti_0 < list_events_[0]) + 1 * (betti_0 >= list_events_[0])
        list_events_[0] = c0

    list_events = [None] + [-nc for nc in list_events_]
    cuts = []
    for i in range(len(list_events_)):
        cuts = cuts + [[tuple(np.sort(edge)) for edge in key_edges_0[list_events[i + 1]:list_events[i]]]]

    tree = Tree(G);

    n_expand = len(list_events_)
    for i in range(n_expand):
        tree.expand(cuts[i], size=min_pixel, proba=entropy_thresh)

    in_segments = [leaf.pixels for leaf in tree.get_leaves()]

    n_kept_pixels = sum([len(leaf_pxl) for leaf_pxl in in_segments])

    n_leaves = len(in_segments)

    n_removed_pixels = len(tree.pixels) - n_kept_pixels
    # print('loss = ', n_removed_pixels, ' pixels | ', round(100. * n_removed_pixels / len(tree.pixels), 2), ' %\n')

    out_segments = [0] * n_removed_pixels
    i = 0
    for cut in tree.out_pixels:
        for seg in cut:
            for pxl in seg:
                out_segments[i] = pxl
                i += 1

    img_labels = np.zeros(img_blurred.shape[:2], dtype='int64')
    height, width = img_blurred.shape[:2]

    pcd_in_segments = [point_cloud_data[seg, :2] for seg in in_segments]
    pcd_out_segments = point_cloud_data[out_segments, :2]

    for l, segment_l in enumerate(pcd_in_segments):
        for pxl in segment_l:
            i = int(pxl[0]);
            j = int(pxl[1]);
            img_labels[i - step_down_i:i + step_up_i + 1, j - step_down_j:j + step_up_j + 1] = l + 1

    distrib = [np.power(np.sum(img_labels == l), -0.12) for l in range(1, np.max(img_labels) + 1)]

    for pxl in pcd_out_segments[::-1, :]:
        i = int(pxl[0]);
        j = int(pxl[1]);
        label_value = []
        k = 1;
        while len(label_value) < 1:
            for delta in [(-2 * k * step_down_i, -2 * k * step_down_j), (-2 * k * step_down_i, 0),
                          (-2 * k * step_down_i, +2 * k * step_up_j), (0, -2 * k * step_down_j),
                          (0, +2 * k * step_up_j), (+2 * k * step_up_i, -2 * k * step_down_j), (+2 * k * step_up_i, 0),
                          (+2 * k * step_up_i, +2 * k * step_up_j)]:
                if 0 <= i + delta[0] < height and width > j + delta[1] >= 0 and \
                        img_labels[i + delta[0], j + delta[1]] != 0.:
                    label_value = label_value + [img_labels[i + delta[0], j + delta[1]]]
                k = k + 1
        if len(set(label_value)) > 1:
            distribution_value = [0] * len(label_value)
            for ind, l in enumerate(label_value):
                distribution_value[ind] = distrib[l - 1]
            distribution_value = distribution_value / sum(distribution_value)

            img_labels[i - step_down_i:i + step_up_i + 1, j - step_down_j:j + step_up_j + 1] = np.random.choice(
                label_value, p=distribution_value)

        else:

            img_labels[i - step_down_i:i + step_up_i + 1, j - step_down_j:j + step_up_j + 1] = label_value[
                0]
        label_value = []

    # remove symmetric expansion
    img_labels = img_labels[:height_0, :width_0]

    return img_blurred, pcd_in_segments, img_labels


def batch_process_images(input_folder, output_folder, ratio):
    total_time = 0.0
    i = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        i += 1
        file_path = os.path.join(input_folder, file_name)
        # 读取图像
        image = cv2.imread(file_path)
        if image is None:
            continue
        image = img_as_float(image)
        super_pixels = image.shape[0] * image.shape[1] * ratio
        start = time.time()
        img_sym, segment_pd, img_labels = pd_segmentation(mode='standard', n_superpixels_=super_pixels, img=image,
                                                          plot_pd=False)
        end = time.time()
        time_piece = end - start
        total_time += time_piece
        my_colors = [(0, 0, 0)] * len(segment_pd)
        for i, seg in enumerate(segment_pd):
            my_colors[i] = tuple(np.random.rand(1, 3)[0])

        save_segmentation_visualization(img_sym, segment_pd, my_colors, output_folder, os.path.splitext(file_name)[0])

    aver_time = total_time / i
    return total_time, aver_time


def save_segmentation_visualization(image, segments, colors, output_path, filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    for i, seg in enumerate(segments):
        plt.plot(seg[:, 1], seg[:, 0], 's', ms=5.1, color=colors[i])
    plt.axis('off')
    plt.savefig(os.path.join(output_path, f"{filename}_segmented.png"))
    plt.close()


if __name__ == '__main__':
    # I = cv2.imread('cityscape/train/img/train1.png')
    # I = img_as_float(I)
    # start = time.time()
    # img_sym, segment_pd, img_labels = pd_segmentation(mode='standard', n_superpixels_=4000, img=I, plot_pd= False)
    # end = time.time()
    # print('Time Consuming: ', end - start)
    #
    # # Create Color List
    # my_colors = [(0, 0, 0)] * len(segment_pd)
    # for i, seg in enumerate(segment_pd):
    #     my_colors[i] = tuple(np.random.rand(1, 3)[0])
    #
    # # Original Image
    # plt.imshow(I)
    # plt.show()
    #
    # plt.imshow(np.ones(img_sym.shape), cmap='gray', vmin=0, vmax=1)
    # for i, seg in enumerate(segment_pd):
    #     plt.plot(seg[:, 1], seg[:, 0], 's', ms=5.1, color=my_colors[i])
    # plt.axis('off')
    # plt.show()

    input_folder = 'cityscape/train/img'
    output_folder = 'segmentation_result_cityscape'
    ratio_list = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
    # ratio_list = [0.12]
    for ratio in ratio_list:
        print('Ratio: ', ratio)
        total_time, aver_time = batch_process_images(input_folder, output_folder, ratio)
        print('Total time: ', total_time)
        print('Average time: ', aver_time)

