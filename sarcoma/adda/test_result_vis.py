from matplotlib import pyplot as plt
import os
from sys import exit as xit
import numpy as np
inited_plot_tests=False
import matplotlib.cm as cm
axs=[None for _ in range(7)]
# copy test-results:
# scp apu:~/FYS-3941/data/test-results.npz ./
def plot_tests(im,y,yhat):
    global inited_plot_tests, axs
    yhat_setmax=1
    yhat[yhat>0.5]=yhat_setmax
    yhat[yhat!=yhat_setmax]=0
    # yhat=np.ones_like(yhat)*yhat_setmax
    yhat = np.ma.masked_where(yhat < 0.5, yhat)
    colors=['red']
    im_cmap=plt.cm.bone
    yhat_kwargs=dict(cmap="Reds_r")
    print(im.shape)
    if not inited_plot_tests:
        # 1) make axes
        # 2) add ax to update
        inited_plot_tests = True
        # Plot image and ground truth
        plt.subplot(1,3,1)
        axs[0]=plt.imshow(im, cmap=plt.cm.bone)
        axs[1]=plt.contour(y, levels=1, colors=["green"])
        # # Plot image and estimated
        plt.subplot(1,3,2)
        # axs[2]=plt.imshow(im, cmap=plt.cm.bone)
        axs[3]=plt.imshow(yhat, **yhat_kwargs)
        # # Plot all three
        plt.subplot(1,3,3)
        axs[4]=plt.imshow(im, cmap=plt.cm.bone)
        axs[5]=plt.contour(y, levels=1, colors=["green"])
        # axs[6]=plt.imshow(yhat*im, **yhat_kwargs)

    else:
        # for i in [0,2]:
        #     axs[i].set_data(im)
        # for i in [3]:
        #     axs[i].set_data(yhat)
        # for i in [1]:
        #     axs[i].ax.collections = []
        #     axs[i].ax.contour(y)
        # # All of them has the image
        for i in [0,4]:
            axs[i].set_data(im)
        for i in [3]:
            axs[i].set_data(yhat)
        for i in [1,5]:
            axs[i].ax.collections = []
            axs[i].ax.contour(y,levels=1, colors=["green"])

    plt.draw()
if __name__ == '__main__':
    # results=np.load("../../mrt2-test-results.npz")
    results=np.load("predictions-Targetmap-validation.npz")
    # X,Y,Yhat,pshape=results['X'],results['Y'],results['Yhat'],results['pshape']
    X,Y,Yhat=results['X'],results['Y'],results['Yhat']
    numinds=np.arange(X.shape[0])
    # print(X.shape,Y.shape,Yhat.shape)
    # np.random.shuffle(numinds)
    for i in numinds:
        xx,yy,yhat=X[i,:,:],Y[i,:,:],Yhat[i,:,:]
        # print(xx.shape)
        plot_tests(xx,yy,yhat)
        plt.pause(0.5)
        # input("Enter for next")
