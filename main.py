import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import PIL.Image as Image

from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as colors

import argparse
# ---------------------------------------------------------------------------------------------------
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]
image_transforms = transforms.Compose([
                        transforms.Resize(pretrained_size),
                        transforms.CenterCrop(pretrained_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                        ])


print("current pid: ", os.getpid())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ---------------------------------------------------------------------------------------------------
def Kmeans(image, h, C, H, W, args):
    print("K-means clustering")
    kmeans = KMeans(n_clusters=args.n_clusters).fit_predict(h)
    kmeans = kmeans.reshape(H, W)

    x = np.arange(H)
    y = np.arange(W)
    fig, ax = plt.subplots()
    # 显示定义colormap，可以用于保持色彩一致
    ax.pcolormesh(x, y, kmeans[:][::-1], cmap='viridis', vmin=0, vmax=args.n_clusters-1)
    plt.savefig(os.path.splitext(image)[0]+'_cluster.jpg')
    plt.cla()
    plt.close(fig)

    return kmeans


def Optics(image, h, C, H, W, args):
    print("OPTICS processing")
    optics = OPTICS(min_samples=5, metric='cosine').fit_predict(h)
    optics = optics.reshape(H, W)

    x = np.arange(H)
    y = np.arange(W)
    fig, ax = plt.subplots()
    # 显示定义colormap，可以用于保持色彩一致
    ax.pcolormesh(x, y, optics[:][::-1], cmap='viridis', vmin=-1, vmax=np.max(optics))
    plt.savefig(os.path.splitext(image)[0]+'_cluster.jpg')
    plt.cla()
    plt.close(fig)

    return optics

def Mean_shift(image, h, C, H, W, args):
    print("Mean shift processing")
    mean_shift = MeanShift(bandwidth=30).fit_predict(h)
    mean_shift = mean_shift.reshape(H, W)

    x = np.arange(H)
    y = np.arange(W)
    fig, ax = plt.subplots()
    # 显示定义colormap，可以用于保持色彩一致
    ax.pcolormesh(x, y, mean_shift[:][::-1], cmap='viridis', vmin=0, vmax=np.max(mean_shift))
    plt.savefig(os.path.splitext(image)[0]+'_cluster.jpg')
    plt.cla()
    plt.close(fig)

    return mean_shift

def Gmm(image, h, C, H, W, args):
    print("Gaussian Mixture Model processing")
    gmm_model = mixture.GaussianMixture(n_components=args.n_clusters, covariance_type='full').fit(h)
    print(gmm_model.means_.shape)
    print(gmm_model.covariances_.shape)
    gmm = gmm_model.predict(h)
    gmm = gmm.reshape(H, W)

    x = np.arange(H)
    y = np.arange(W)
    fig, ax = plt.subplots()
    # 显示定义colormap，可以用于保持色彩一致
    ax.pcolormesh(x, y, gmm[:][::-1], cmap='viridis', vmin=0, vmax=args.n_clusters-1)
    plt.savefig(os.path.splitext(image)[0]+'_cluster.jpg')
    plt.cla()
    plt.close(fig)

    return gmm

# ---------------------------------------------------------------------------------------------------
def tsne(image, h, C, H, W, args, cluster = None):
    # FIXME: may have bugs
    if args.tsne == '1d':
        pca = PCA(n_components=args.tsne_dim)
        h = pca.fit_transform(h)
        tsne = TSNE(n_components=1, perplexity = args.perplexity, early_exaggeration=args.early_exaggeration, metric=args.metric, learning_rate=100, n_iter=2000)
        embedding = tsne.fit_transform(h)
        
        fig, ax = plt.subplots()
        if cluster.all() == None:
            for i in range(h.shape[0]):
                ax.scatter(embedding[i][0], 0)
        else:
            vmax = np.max(cluster)
            cluster = np.reshape(cluster, H*W)
            for i in range(h.shape[0]):
                # TODO: change into density map
                ax.scatter(embedding[i][0], 0, c=cluster[i], cmap='viridis', vmin=0, vmax=vmax, marker='|')
            plt.savefig(os.path.splitext(image)[0]+'_tsne.jpg')
            plt.cla()
            plt.close(fig)


    if args.tsne == '2d':
        pca = PCA(n_components=args.tsne_dim)
        h = pca.fit_transform(h)
        tsne = TSNE(n_components=2, perplexity = args.perplexity, early_exaggeration=args.early_exaggeration, metric=args.metric, learning_rate=100, n_iter=2000)
        embedding = tsne.fit_transform(h)
        fig, ax = plt.subplots()
        if cluster.all() == None:
            for i in range(h.shape[0]):
                ax.scatter(embedding[i][0], embedding[i][1])
        else:
            vmax = np.max(cluster)
            cluster = np.reshape(cluster, H*W)
            for i in range(h.shape[0]):
                ax.scatter(embedding[i][0], embedding[i][1], c=cluster[i], cmap='viridis', vmin=0, vmax=vmax)
            plt.savefig(os.path.splitext(image)[0]+'_tsne.jpg')
            plt.cla()
            plt.close(fig)


    if args.tsne == '3d':
        pca = PCA(n_components=args.tsne_dim)
        h = pca.fit_transform(h)
        tsne = TSNE(n_components=3, perplexity = args.perplexity, early_exaggeration=args.early_exaggeration, metric=args.metric, learning_rate=100, n_iter=2000)
        embedding = tsne.fit_transform(h)
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig)
        print(h.shape)
        print(embedding.shape)

        if cluster.all() == None:
            for i in range(h.shape[0]):
                ax.scatter(embedding[i][0], embedding[i][1], embedding[i][2])
        else:
            vmax = np.max(cluster) 
            cluster = np.reshape(cluster, H*W)
            print(cluster.shape)
            for i in range(h.shape[0]):
                #显示定义scatter的colormap
                ax.scatter(embedding[i][0], embedding[i][1], embedding[i][2], c=cluster[i], cmap='viridis' , vmin=0, vmax=vmax)
        def rotate(angle):
            ax.view_init(azim=angle)
        angle = 3
        ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
        ani.save(os.path.splitext(image)[0]+'_tsne.gif', writer=animation.PillowWriter(fps=20))

        plt.cla()
        plt.close(fig)
        
# ---------------------------------------------------------------------------------------------------
# take vgg19_relu_4_1 as example
def process_image_file(args):
    model = models.vgg19(pretrained=True).features[:21]

    image_path = os.listdir(args.image_root)
    for image in image_path:
        print("Now processing: ", image)
        x = Image.open(args.image_root + image)
        x = image_transforms(x)
        x = torch.unsqueeze(x, dim=0)

        with torch.no_grad():
            h = model(x)
            h = torch.squeeze(h, dim=0)
            C, H, W = h.shape
            h = torch.reshape(h, (C, H*W)).numpy()
            h = np.transpose(h)
            if args.cluster == 'kmeans':
                cluster = Kmeans(image, h, C, H, W, args)
            elif args.cluster == 'optics':
                cluster = Optics(image, h, C, H, W, args)
            elif args.cluster == 'mean_shift':
                cluster = Mean_shift(image, h, C, H, W, args)
            elif args.cluster == 'gmm':
                cluster = Gmm(image, h, C, H, W, args)
            elif args.cluster == 'none':
                cluster = None
            
            if args.tsne != 'none':
                tsne(image, h, C, H, W, args, cluster)
# ---------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="parser for latent space analysis")
    parser.add_argument('--image_root', type=str, default='/Users/LX/Desktop/latent-space-analysis/clustering/temp/', help='the path of image folder')

    #clustering
    parser.add_argument('--cluster', type=str, default='none', choices=['none', 'kmeans', 'optics', 'mean_shift', 'gmm'], help='the algorithm for clustering')
    parser.add_argument('--n_clusters', type=int, default=2, help='predefined k for k-means or other clustering algorithms')

    #t-SNE: other parameters need to be adjusted manually
    parser.add_argument('--tsne', type=str, default='none', choices=['none', '1d', '2d', '3d'], help='whether or not to use t-SNE-3d for visualization')
    parser.add_argument('--tsne_dim', type=int, default=128, help='the dimension reducing to this value before applying t-SNE')
    parser.add_argument('--perplexity', type=int, default=50, help='perplexity for t-SNE')
    parser.add_argument('--early_exaggeration', type=int, default=1, help='episilon for t-SNE')
    parser.add_argument('--metric', type=str, default='cosine', help='distance definition for t-SNE')
    args = parser.parse_args()

    # run analysis
    process_image_file(args)
# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()