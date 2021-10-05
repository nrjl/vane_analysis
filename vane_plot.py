import os
import h5py
import matplotlib.pyplot as plt
import argparse
import numpy as np


class PlotItem(object):
    def __init__(self, name, items, multiplier=1.0,
                 labels=None):
        self.name = name
        self.items = items
        if isinstance(multiplier, float):
            self.multiplier = multiplier*np.ones(len(self.items))
        else:
            assert len(multiplier) == len(self.items)
            self.multiplier = multiplier

        if labels is None:
            self.labels = ['{0}*{1:.4f}'.format(i, m) for i, m in zip(self.items, self.multiplier)]
        else:
            self.labels = labels
        assert len(self.labels) == len(self.items), print('len(labels) != len(items)')


    def add_to_plot(self, axis, rec, time):
        y = np.hstack([to_numpy(rec, k) for k in self.items]) * self.multiplier
        h_line = axis.plot(time, y)
        axis.set_ylabel(self.name)
        axis.legend(h_line, self.labels)
        return h_line


def to_numpy(rec, key):
    # Scrappy wrapper
    return np.array(rec[key]).T


def plot_records(h5_file, rec_name='11_yaw_sweep_aoa_0', items=None):
    if items is None:
        items = [PlotItem('Velocity [m/s]',
                          ['airspeed_0.true_airspeed_m_s', 'wt.vel'],
                          1.0)]

    # Read hdf5 file
    fh = h5py.File(h5_file, "r")
    rec = fh[rec_name]
    time = to_numpy(rec, 'Time')

    # Make plots
    fig, ax = plt.subplots(len(items), 1)
    fig.set_size_inches([9,7])
    if len(items) == 1:
        ax = [ax]

    # Plot each item
    h_lines = []
    for a, item in zip(ax, items):
        h_lines.append(item.add_to_plot(a, rec, time))

    ax[-1].set_xlabel('Timeframe in [s]')
    ax[0].set_title(rec_name)

    fh.close()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot vane offsets from wind tunnel data')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Plot all records')
    parser.add_argument('base_dir', type=str, help='Base directory of deep stall data')
    args = parser.parse_args()

    h5file = os.path.join(args.base_dir, 'Raw_Data_Set', 'Wind_Tunnel', 'WT_raw.h5')

    # Generate a PlotItem for each thing you want to plot. You need to specify
    # the name, the list of item fields, a multiplier (for units) and the labels
    plot_items = [
        PlotItem('Velocity [m/s]', ['airspeed_0.indicated_airspeed_m_s', 'wt.vel'],
                 1.0),
        PlotItem('Alpha [deg]', ['airflow_aoa_0.aoa_rad', 'wt.aoa'],
                 180.0/np.pi),
        PlotItem('Beta [deg]', ['airflow_slip_0.slip_rad', 'wt.aos'],
                 180.0/np.pi)]

    if args.all:
        f = h5py.File(h5file, "r")
        record_names = list(f.keys())
        f.close()
        for rec_name in record_names:
            plot_records(h5file, rec_name, plot_items)
    else:
        plot_records(h5file, '11_yaw_sweep_aoa_0', plot_items)
        plot_records(h5file, '11_yaw_sweep_aoa_5', plot_items)
        plot_records(h5file, '11_aoa_sweep_m5', plot_items)
        plot_records(h5file, '15_yaw_sweep_aoa_0', plot_items)

    plt.show()
