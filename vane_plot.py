import os
import h5py
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


class PlotItem(object):
    def __init__(self, name, items, multiplier=1.0,
                 labels=None):
        self.name = name
        self.items = items

        # We want a multiplier for each item (if only one, assume all equal)
        self.multiplier = np.atleast_1d(multiplier)
        if len(self.multiplier) == 1:
            self.multiplier = self.multiplier*np.ones(len(self.items))
        else:
            assert len(multiplier) == len(self.items)
            self.multiplier = multiplier

        if labels is None:
            self.labels = ['{0}*{1:.4f}'.format(i, m)
                           for i, m in zip(self.items, self.multiplier)]
        else:
            self.labels = labels
        assert len(self.labels) == len(self.items), \
            print('len(labels) != len(items)')

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

def plot_bias(h5_file, rec_name = [], fields = [], skip_data=1, aoa_limit=100, aos_limit=100, fit_bias=False):
    skip_data = int(skip_data)
    if skip_data < 1:
        skip_data = 1

    # get the data
    fh = h5py.File(h5_file, "r")
    data = {}
    for f in fields:
        data[f] = []
    data['aoa_meas'] = []
    data['wt.aoa'] = []
    data['aos_meas'] = []
    data['wt.aos'] = []
    data['time'] = []
    data['q0'] = []
    data['q1'] = []
    data['q2'] = []
    data['q3'] = []
    data['actuator_outputs_0.Throttle'] = []

    for name in rec_name:
        if max(np.squeeze(fh[name]['wt.aoa'][::].T[::skip_data]).tolist()) < 2 * np.pi:
            data['aoa_meas'] += np.rad2deg(np.squeeze(fh[name]['airflow_aoa_0.aoa_rad'][::].T[::skip_data])).tolist()
            data['wt.aoa'] += np.rad2deg(np.squeeze(fh[name]['wt.aoa'][::].T[::skip_data])).tolist()
            data['aos_meas'] += np.rad2deg(np.squeeze(fh[name]['airflow_slip_0.slip_rad'][::].T[::skip_data])).tolist()
            data['wt.aos'] += np.rad2deg(-np.squeeze(fh[name]['wt.aos'][::].T[::skip_data])).tolist()
            data['time'] += np.squeeze(fh[name]['Time'][::].T[::skip_data]).tolist()
            data['q0'] += np.squeeze(fh[name]['vehicle_attitude_0.q_0'][::].T[::skip_data]).tolist()
            data['q1'] += np.squeeze(fh[name]['vehicle_attitude_0.q_1'][::].T[::skip_data]).tolist()
            data['q2'] += np.squeeze(fh[name]['vehicle_attitude_0.q_2'][::].T[::skip_data]).tolist()
            data['q3'] += np.squeeze(fh[name]['vehicle_attitude_0.q_3'][::].T[::skip_data]).tolist()
            data['actuator_outputs_0.Throttle'] += np.squeeze(fh[name]['actuator_outputs_0.Throttle'][::].T[::skip_data]).tolist()
 
            for f in fields:
                if not f in ['aoa_meas', 'aos_meas', 'wt.aoa', 'wt.aos', 'roll', 'pitch', 'yaw', 'actuator_outputs_0.Throttle']:
                    data[f] += np.squeeze(fh[name][f][::].T[::skip_data]).tolist()

    # convert to np.array
    for k in data.keys():
        data[k] = np.array(data[k])

    # mask by aoa/aos if requested
    if aoa_limit > 0:
        mask = np.abs(data['wt.aoa']) < aoa_limit
        for k in data.keys():
            data[k] = data[k][mask]

    throttle_limit = 500
    if throttle_limit > 0:
        mask = np.abs(data['actuator_outputs_0.Throttle']) > throttle_limit
        for k in data.keys():
            data[k] = data[k][mask]

    if aos_limit > 0:
        mask = np.abs(data['wt.aos']) < aos_limit
        for k in data.keys():
            data[k] = data[k][mask]

    # compute the euler angles
    rotations = R.from_quat(np.stack([data['q0'],data['q1'],data['q2'],data['q3']], axis=1))
    euler_angles = rotations.as_euler('zyx',True)
    data['yaw'] = euler_angles[:,0]
    data['pitch'] = euler_angles[:,1]
    data['roll'] = euler_angles[:,2]

    # aoa plots
    aoa_bias_deg = data['aoa_meas'] - data['wt.aoa']
    fig, ax = plt.subplots(len(fields), 1)
    fig.set_size_inches([9,7])
    if len(fields) == 1:
        ax = [ax]

    for a, f in zip(ax, fields):
        a.scatter(data[f], aoa_bias_deg, c = data['airspeed_0.true_airspeed_m_s'])
        a.set_xlabel(f)
        a.set_ylabel('aoa bias [deg]')

    # aoa plots
    aos_bias_deg = data['aos_meas'] - data['wt.aos']
    fig, ax = plt.subplots(len(fields), 1)
    fig.set_size_inches([9,7])
    if len(fields) == 1:
        ax = [ax]

    for a, f in zip(ax, fields):
        a.scatter(data[f], aos_bias_deg, c = data['airspeed_0.true_airspeed_m_s'])
        a.set_xlabel(f)
        a.set_ylabel('aos bias [deg]')

    if fit_bias:
        from scipy.optimize import minimize
        def aoa_bias(p):
            return p[0] + p[1] * data['aoa_meas'] + p[2] * (data['aoa_meas'] + p[3]) * (data['airspeed_0.true_airspeed_m_s'] + p[4]) + p[5] * data['actuator_outputs_0.Throttle']

        def aoa_error(p):
            return np.sum((aoa_bias(p) - aoa_bias_deg)**2)

        res_aoa = minimize(aoa_error, [0,0,0,0,0,0], method='nelder-mead',
                           options={'xatol': 1e-8, 'disp': True})

        aoa_bias_est = aoa_bias(res_aoa.x)
        aoa_est = data['aoa_meas'] - aoa_bias_est
        
        aoa_bias_error = aoa_bias(res_aoa.x) - aoa_bias_deg
        fig, ax = plt.subplots(len(fields), 1)
        fig.set_size_inches([9,7])
        if len(fields) == 1:
            ax = [ax]

        for a, f in zip(ax, fields):
            a.scatter(data[f], aoa_bias_est, c = data['airspeed_0.true_airspeed_m_s'])
            a.set_xlabel(f)
            a.set_ylabel('aoa bias error [deg]')

        def aos_bias(p):
            return p[0] + p[1] * (p[2] * data['airspeed_0.true_airspeed_m_s'] - p[3]) * (1+np.tanh(p[4] * (data['aoa_meas'] - p[5]))) + p[6] * data['aos_meas']
        
        def aos_error(p):
            return np.sum((aos_bias(p) - aos_bias_deg)**2)

        res_aos = minimize(aos_error, [0,0,1,4,1,0,0], method='cg',
                           options={'xatol': 1e-8, 'disp': True, 'maxiter': 10000, 'maxfev': 10000})

        aos_bias_est = aos_bias(res_aos.x)
        aos_bias_error = aos_bias_est - aos_bias_deg
        fig, ax = plt.subplots(len(fields), 1)
        fig.set_size_inches([9,7])
        if len(fields) == 1:
            ax = [ax]

        for a, f in zip(ax, fields):
            a.scatter(data[f], aos_bias_est, c = data['airspeed_0.true_airspeed_m_s'])
            a.set_xlabel(f)
            a.set_ylabel('aos bias error [deg]')

        print(res_aoa.x)
        print(res_aos.x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot vane offsets from wind tunnel data')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Plot all records')
    parser.add_argument('-c', '--cal', action='store_true',
                        help='Show calibrated data')
    parser.add_argument('-b', '--bias', action='store_true',
                        help='Show the bias plots')
    parser.add_argument('-e', '--experiments', action='store_true',
                        help='Plot all the different experiment names')
    parser.add_argument('base_dir', type=str,
                        help='Base directory of deep stall data')
    parser.add_argument('-la', '--aoa_limit', default=100, type=float,
                        help='Limit of the angle of attack plotted')
    parser.add_argument('-ls', '--aos_limit', default=100, type=float,
                        help='Limit of the angle of slip plotted')
    parser.add_argument('-f', '--fit_bias', action='store_true',
                        help='Fit the bias and plot the result')
    args = parser.parse_args()

    # Generate a PlotItem for each thing you want to plot. You need to specify:
    # - name
    # - list of item fields
    # - multiplier (for units, can be an array with value for each, optional)
    # - labels (optional)
    plot_items = [
        PlotItem('Velocity [m/s]',
                 ['airspeed_0.indicated_airspeed_m_s', 'wt.vel'],
                 1.0),
        PlotItem('Alpha [deg]',
                 ['airflow_aoa_0.aoa_rad', 'wt.aoa'],
                 180.0/np.pi),
        PlotItem('Beta [deg]',
                 ['airflow_slip_0.slip_rad', 'wt.aos'],
                 180.0/np.pi)]

    if args.cal:
        h5file = os.path.join(args.base_dir, 'Cal_Data_Set', 'Wind_Tunnel',
                              'WT_cal.h5')
    else:
        h5file = os.path.join(args.base_dir, 'Raw_Data_Set', 'Wind_Tunnel',
                              'WT_raw.h5')
        # Sign is flipped for slip sensor
        plot_items[2].multiplier = 180.0/np.pi*np.array([-1, 1])

    if args.experiments:
        f = h5py.File(h5file, "r")
        record_names = list(f.keys())
        print(record_names)
        f.close()

    if  args.bias:
        if args.all:
            f = h5py.File(h5file, "r")
            record_names = list(f.keys())
            f.close()
        else:
            record_names = ['11_aoa_sweep_m5',
                            '11_motor_aoa_0',
                            '11_motor_aoa_10',
                            '11_motor_aoa_12',
                            '11_motor_aoa_14',
                            '11_motor_aoa_16',
                            '11_motor_aoa_20',
                            '11_motor_aoa_5',
                            '11_motor_aoa_8',
                            '11_motor_aoa_m5',
                            '11_motor_aoa_sweep_m5',
                            '11_yaw_sweep_aoa_0',
                            '11_yaw_sweep_aoa_10',
                            '11_yaw_sweep_aoa_12',
                            '11_yaw_sweep_aoa_14',
                            '11_yaw_sweep_aoa_16',
                            '11_yaw_sweep_aoa_20',
                            '11_yaw_sweep_aoa_5',
                            '11_yaw_sweep_aoa_8',
                            '11_yaw_sweep_aoa_m5']

        plot_bias(h5file, record_names, ['wt.aoa', 'wt.aos', 'aoa_meas', 'aos_meas', 'airspeed_0.true_airspeed_m_s'], 40, args.aoa_limit, args.aos_limit, args.fit_bias)

    else:
        if args.all:
            f = h5py.File(h5file, "r")
            record_names = list(f.keys())       # Also useful to see all records
            f.close()
            for rec_name in record_names:
                plot_records(h5file, rec_name, plot_items)
        else:
            plot_records(h5file, '11_yaw_sweep_aoa_0', plot_items)
            plot_records(h5file, '11_yaw_sweep_aoa_5', plot_items)
            plot_records(h5file, '11_aoa_sweep_m5', plot_items)
            plot_records(h5file, '15_yaw_sweep_aoa_0', plot_items)

    plt.show()
