# -*- coding: utf-8 -*-
"""Scalogram gradient-based reconstruction on exponential chirp GIF.

  - Differentiability currently requires GPU
    (PyTorch path only runs w/ `os.environ['SSQ_GPU'] = '1'`)
  - `ssq_cwt` in the GIF is for visualization only; only `cwt` has
    differentiability implemented

mp4 -> gif:
    ffmpeg -i reconstruction.mp4 -vf
    "fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" reconstruction.gif

    followed by https://www.freeconvert.com/gif-compressor
    compression level 10, number of colors 64
"""
import os
import numpy as np
import torch
from ssqueezepy import cwt, ssq_cwt, Wavelet, TestSignals
from ssqueezepy.visuals import plot

os.environ['SSQ_GPU'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def rel_l2(x0, x1):
    return _rel_dist(x0, x1, torch.linalg.norm)

def rel_l1(x0, x1):
    return _rel_dist(x0, x1, lambda x: torch.linalg.norm(x, ord=1))

def _rel_dist(x0, x1, norm_fn):
    return norm_fn(x0 - x1) / (norm_fn(x0) + 1e-7)

#%% Prepare transform & signal ###############################################
N = 2048
wavelet = Wavelet(N=N)
xorig = TestSignals(N=N).echirp(fmin=8)[0]

_, scales = cwt(torch.ones(N), wavelet)  # reusing `scales` is faster

def transform(x):
    return torch.abs(cwt(x, wavelet, scales=scales)[0])

#%%# Configure optimization ##################################################
n_iters = 100
loss_switch_iter = n_iters  # var reused from another example where != n_iters
y = torch.from_numpy(xorig).to(device)
Sy = transform(y)
div = Sy.std()
Sy /= div

torch.manual_seed(1)
x = torch.randn(N, device=device)
x /= torch.max(torch.abs(x))
x.requires_grad = True
optimizer = torch.optim.Adam([x], lr=.8)
loss_fn = torch.nn.MSELoss()
dist_fn = rel_l2

#%% Run optimization #########################################################
losses, losses_recon = [], []
x_recons = []
lrs = []
for i in range(n_iters):
    optimizer.zero_grad()
    Sx = transform(x)
    Sx /= div
    loss = loss_fn(Sx, Sy)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu().numpy()))
    xn, yn = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    losses_recon.append(float(dist_fn(y, x)))
    x_recons.append(xn)

#%% Plot & print end results
end_ratio = losses[0] / losses[-1]

kw = dict(show=1, abs=1)
plot(np.log10(losses), **kw, title="log10(losses)")
plot(np.log10(losses_recon), **kw, title="log10(losses_recon)")
plot(xn, show=1)

print(("\nReconstruction (torch):\n(end_start_ratio, min_loss, "
       "min_loss_recon) = ({:.1f}, {:.2e}, {:.6f})").format(
           end_ratio, min(losses), min(losses_recon)))

#%% take SSQ of every reconstruction
wavelet = Wavelet(('gmw', {'gamma': 1, 'beta': 1}))
x_recons = np.array(x_recons)
ssq_x_recon = torch.abs(ssq_cwt(x_recons, wavelet, scales='log')[0]).cpu().numpy()

#%%# Animate #################################################################
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PlotImshowAnimation(animation.TimedAnimation):
    def __init__(self, imshow_frames, plot_frames0, plot_frames1, plot_frames2):
        self.imshow_frames = imshow_frames
        self.plot_frames0 = plot_frames0
        self.plot_frames1 = plot_frames1
        self.plot_frames2 = plot_frames2
        self.xticks = np.arange(len(plot_frames1))

        self.n_repeats_total = n_repeats * repeat_first
        self.n_frames = (len(imshow_frames) + self.n_repeats_total - repeat_first
                         + repeat_last)

        self.title_kw = dict(weight='bold', fontsize=15, loc='left')
        self.label_kw = dict(weight='bold', fontsize=15, labelpad=3)
        self.txt_kw = dict(x=0, y=1.017, s="", ha="left", weight='bold')
        ticks_labelsize = 14

        fig, axes = plt.subplots(2, 2, figsize=(18/1.5, 9))

        # plots ##############################################################
        ax = axes[0, 0]
        ax.plot(self.plot_frames0[0]*1.03)
        ax.set_xlim(-30, 2078)
        ax.set_title("x_reconstructed", **self.title_kw)
        ax.set_yticks([-1, -.5, 0, .5, 1])
        ax.set_yticklabels([r'$\endash 1.0$', r'$\endash 0.5$', '0',
                            '0.5', '1.0'])
        self.lines0 = [ax.lines[-1]]
        ax.tick_params(labelsize=ticks_labelsize)

        # imshows ############################################################
        ax = axes[0, 1]
        # color norm
        mx = np.max(imshow_frames) * .5
        im = ax.imshow(self.imshow_frames[0], cmap='turbo', animated=True,
                       aspect='auto', vmin=0, vmax=mx)
        self.ims1 = [im]
        ax.set_title("|ssq_cwt(x_reconstructed)|", **self.title_kw)
        ax.set_yticks([])
        ax.tick_params(labelsize=ticks_labelsize)

        # plots ##############################################################
        ax = axes[1, 0]
        ax.plot(self.xticks, self.plot_frames1)
        ax.set_xlabel("n_iters", **self.label_kw)
        ax.set_yticks([0, -1, -2, -3])
        ax.set_yticklabels(['0'] + [rf'$\endash {n}$' for n in (1, 2, 3)])
        ax.set_ylim([-3, 0])
        ax.tick_params(labelsize=ticks_labelsize)

        self.lines2 = [ax.lines[-1]]
        self.lines2[0].set_data([self.xticks[0]], [self.plot_frames1[0]])
        self.txt2 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        ax = axes[1, 1]
        ax.plot(self.xticks, self.plot_frames2)
        ax.set_xlabel("n_iters", **self.label_kw)
        ax.set_yticks([])
        ax.tick_params(labelsize=ticks_labelsize)

        self.lines3 = [ax.lines[-1]]
        self.lines3[0].set_data([self.xticks[0]], [self.plot_frames2[0]])
        self.txt3 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        # finalize #######################################################
        fig.subplots_adjust(left=.048, right=.985, bottom=.068, top=.96,
                            wspace=.025, hspace=.15)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        if frame_idx == 0:
            self.loss_idx = 0
            self.prev_loss_idx = 0
        elif frame_idx % n_repeats == 0 or frame_idx > self.n_repeats_total:
            N = len(self.imshow_frames)
            tail_start = max(0, N - repeat_last)

            if frame_idx > self.n_repeats_total and self.loss_idx >= tail_start:
                if self.loss_idx < N - 1:
                    self.loss_idx += 1
                else:
                    # Already at final frame: keep holding.
                    return
            else:
                self.loss_idx += 1

        if self.loss_idx == self.prev_loss_idx:
            return
        self.prev_loss_idx = self.loss_idx
        frame_idx = self.loss_idx  # adjusted

        self.lines0[0].set_ydata(self.plot_frames0[frame_idx])
        self.ims1[0].set_array(self.imshow_frames[ frame_idx])
        self.lines2[0].set_data(self.xticks[:frame_idx],
                                self.plot_frames1[:frame_idx])
        self.lines3[0].set_data(self.xticks[:frame_idx],
                                self.plot_frames2[:frame_idx])

        loss = self.plot_frames1[frame_idx]
        loss_recon = self.plot_frames2[frame_idx]
        txt = "log10(loss_scalogram)={:.1f} ({})".format(loss, "L2")
        self.txt2.set_text(txt)
        self.txt3.set_text("log10(loss_x_reconstructed)={:.1f}".format(
            loss_recon))

        # finalize ###########################################################
        self._drawn_artists = [*self.lines0, *self.ims1, *self.lines2,
                               *self.lines3, self.txt2, self.txt3]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

losses, losses_recon = [ls.copy() for ls in (losses, losses_recon)]
imshow_frames = ssq_x_recon
plot_frames0 = x_recons
plot_frames1 = np.log10(losses)
plot_frames2 = np.log10(losses_recon)

repeat_first = 1
repeat_last = 10
n_repeats = 5

ani = PlotImshowAnimation(imshow_frames, plot_frames0, plot_frames1, plot_frames2)
ani.save('reconstruction.mp4', fps=10)
plt.show()
