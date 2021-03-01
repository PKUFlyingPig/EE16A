from __future__ import division
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# from IPython.html.widgets import interact, interactive    # use this for old lab computer build
from ipywidgets import interact, interactive                # use this for newer ipython version
from IPython.display import display
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt
from random import random
import scipy.io
import scipy.signal

# Constants
v = 340.29
sampling_rate = 44100
carrier_freq = 12000
up_sample_rate = 20
beacon_length = 511

# The location of the speakers
speakers = [(0, 0), (5, 0), (0, 5), (5, 5), (0, 2.5), (2.5, 0)]
x0, y0 = 0, 0
x1, y1 = 5, 0
x2, y2 = 0, 5
x3, y3 = 5, 5

RANDOM_OFFSET = True # Set to False so the signals are in phase.


###########
# SIGNALS #
###########

MATLAB_FILE_NAME = 'support_code/data.mat'

class Signal(object):
	def __init__(self):
		self.mat_file = scipy.io.loadmat(MATLAB_FILE_NAME)

	def get_variable(self, var_name):
		if var_name in self.mat_file:
			return self.mat_file[var_name]
		return None

# singleton
Signal = Signal()

LPF = [
		 0.011961120715177888, 0.017898431488832207,
		 0.023671088777712977, 0.023218234511504027,
	     0.014273841944786474, -0.0019474257432360068,
	     -0.020353564773665882, -0.033621739954825786,
	     -0.03551208496295946, -0.02427977087856695,
	     -0.004505671193971414, 0.01417985329617787,
	     0.02120515509832614, 0.010524501050778973,
	     -0.01526011832898763, -0.044406684807689674,
	     -0.06003132747249487, -0.047284818566536956,
	     -0.0006384172460392303, 0.07212437221154776,
	     0.151425957854483, 0.21267819920402747,
	     0.23568789602698456, 0.21267819920402747,
	     0.151425957854483, 0.07212437221154776,
	     -0.0006384172460392303, -0.047284818566536956,
	     -0.06003132747249487, -0.044406684807689674,
	     -0.01526011832898763, 0.010524501050778973,
	     0.02120515509832614, 0.01417985329617787,
	     -0.004505671193971414, -0.02427977087856695,
	     -0.03551208496295946, -0.033621739954825786,
	     -0.020353564773665882, -0.0019474257432360068,
	     0.014273841944786474, 0.023218234511504027,
	     0.023671088777712977, 0.017898431488832207,
	     0.011961120715177888
	   ]


beacon0 = Signal.get_variable("beacon0")[0]
beacon1 = Signal.get_variable("beacon1")[0]
beacon2 = Signal.get_variable("beacon2")[0]
beacon3 = Signal.get_variable("beacon3")[0]
beacon4 = Signal.get_variable("beacon4")[0]
beacon5 = Signal.get_variable("beacon5")[0]

beacon = [beacon0, beacon1, beacon2, beacon3, beacon4, beacon5]
signal_length = len(beacon0)

def cross_correlation(stationary_signal, sliding_signal):
    """Compute the cross_correlation of two given signals
    Args:
    stationary_signal (np.array): input signal 1
    sliding_signal (np.array): input signal 2

    Returns:
    cross_correlation (np.array): cross-correlation of stationary_signal and sliding_signal

    >>> cross_correlation([0, 1, 2, 3], [0, 2, 3, 0])
    [8, 13, 6, 3]
    """
    # new "infinitely periodic correletaion" using np.correlate like in HW
    inf_stationary_signal = np.concatenate((stationary_signal,stationary_signal))
    entire_corr_vec = np.correlate(inf_stationary_signal, sliding_signal, 'full')
    return entire_corr_vec[len(sliding_signal)-1: len(sliding_signal)-1 + len(sliding_signal)]
    # old implementation
    # return np.fft.ifft(np.fft.fft(stationary_signal) * np.fft.fft(sliding_signal).conj()).real

def cross_corr_demo_1():
    # Input signals for which to compute the cross-correlation
    signal1 = np.array([1, 2, 3, 2, 1, 0]) #sliding
    signal2 = np.array([3, 1, 0, 0, 0, 1]) #inf periodic stationary
    print('input stationary_signal: '+str(signal2))
    print('input sliding_signal: '+str(signal1))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal2, np.roll(signal1,k))[0] for k in range(len(signal2))]
    print( 'cross-correlation:'+str(corr))

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(12,6))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal2, 'rx-', label='stationary')
        plt.plot(signal1, 'bo-', label='sliding')
        plt.xlim(0, 5)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        plt.title('Computed cross-correlation(%d)=%d\n%s\n%s'%(i, np.dot(signal1, signal2), str(signal2), str(signal1)))
        signal2 = np.roll(signal2, -1)

    # Adjust subplot spacing
    #plt.tight_layout()
    plt.figure()
    plt.plot(corr,'ko-')
    plt.xlim(0, len(signal2)-1)
    plt.ylim(0, 15)
    plt.title('Cross-correlation (single-period)')

def cross_corr_demo_2():
    # Here we repeat the above example for a two-period case

    # Input signals for which to compute the cross-correlation
    # Make signals periodic with the numpy.tile function
    Nrepeat = 2
    signal1 = np.array([1, 2, 3, 2, 1, 0])
    signal1 = np.tile(signal1, Nrepeat)
    signal2 = np.array([3, 1, 0, 0, 0, 1])
    signal2 = np.tile(signal2, Nrepeat)
    print('input stationary signal: '+str(signal2))
    print('input sliding signal: '+str(signal1))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal2, np.roll(signal1,k))[0] for k in range(len(signal2))]
    print( 'cross-correlation:'+str(corr))

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(12,12))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal1, 'bo-', label='sliding')
        plt.plot(signal2, 'rx-', label='stationary')
        plt.xlim(0, 11)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        plt.title('Computed cross-correlation(%d)=%d\n%s\n%s'%(i, np.dot(signal1, signal2), str(signal2), str(signal1)))
        signal2 = np.roll(signal2, -1)

    # Adjust subplot spacing
    #plt.tight_layout()
    plt.figure()
    plt.plot(corr,'ko-')
    plt.xlim(0, 11)
    plt.ylim(0, 28)
    plt.title('Cross-correlation (two-period)')

def test_correlation_plot(signal1, signal2, lib_result, your_result):
    # Plot the output
    fig = plt.figure(figsize=(8,3))
    ax = plt.subplot(111)
    str_corr='Correct Answer (length='+str(len(lib_result))+')'
    str_your='Your Answer (length='+str(len(your_result))+')'

    ax.plot([x-len(signal2)+1 for x in range(len(lib_result))], lib_result, 'k', label=str_corr, lw=1.5)
    ax.plot([x-len(signal2)+1 for x in range(len(your_result))], your_result, '--r', label=str_your, lw = 3)
    ax.set_title("Cross correlation of:\n%s\n%s"%(str(signal1), str(signal2)))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def cross_corr_test():
    # You can change these signals to get more test cases
    # Test 1
    signal1 = np.array([1, 5, 8, 6])
    signal2 = np.array([1, 3, 5, 2])

    # Run the test
    lib_result, your_result = test_correlation(cross_correlation, signal1, signal2)
    test_correlation_plot(signal1, signal2, lib_result, your_result)

    # Test 2
    signal1 = np.array([1, 5, 8, 6, 1, 5, 8, 6])
    signal2 = np.array([1, 3, 5, 2, 1, 3, 5, 2])

    # Run the test
    lib_result, your_result = test_correlation(cross_correlation, signal1, signal2)
    test_correlation_plot(signal1, signal2, lib_result, your_result)

    # Test 3
    signal1 = np.array([1, 3, 5, 2])
    signal2 = np.array([1, 5, 8, 6])

    # Run the test
    lib_result, your_result = test_correlation(cross_correlation, signal1, signal2)
    test_correlation_plot(signal1, signal2, lib_result, your_result)

def demodulate_signal(signal):
	"""
	Demodulate the signal using complex demodulation.
	"""
	# Demodulate the signal using cosine and sine bases
	demod_real_base = [cos(2 * pi * carrier_freq * i / sampling_rate)
		for i in range(1, len(signal) + 1)]
	demod_imaginary_base = [sin(2 * pi * carrier_freq * i / sampling_rate)
		for i in range(1, len(signal) + 1)]
	# Multiply the bases to the signal received
	demod_real = [demod_real_base[i] * signal[i] for i in range(len(signal))]
	demod_imaginary = [demod_imaginary_base[i] * signal[i] for i in range(len(signal))]
	# Filter the signals
	demod_real = np.convolve(demod_real, LPF)
	demod_imaginary = np.convolve(demod_imaginary, LPF)

	return np.asarray(demod_real) + np.asarray(demod_imaginary * 1j)

def average_signal(signal):
	beacon_length = len(beacon[0])
	num_repeat = len(signal) // beacon_length
	signal_reshaped = signal[0 : num_repeat * beacon_length].reshape((num_repeat, beacon_length))
	averaged = np.mean(np.abs(signal_reshaped), 0).tolist()
	return averaged

def generate_carrier_with_random_offset():
	rand = random()
	carrier_sample = (2 * pi *
		(carrier_freq * sample / sampling_rate + rand)
		for sample in range(1, signal_length + 1))
	return [cos(sample) for sample in carrier_sample]

def generate_carrier():
	carrier_sample = (2 * pi *
		carrier_freq * sample / sampling_rate
		for sample in range(1, signal_length + 1))
	return [cos(sample) for sample in carrier_sample]

def modulate_signal(signal, carrier):
	"""
	Modulate a given signal. The length of both signals MUST
	be the same.
	"""
	return [signal[i] * carrier[i] for i in range(signal_length)]

# Modulate beacon signals
if RANDOM_OFFSET:
        modulated_beacon0 = modulate_signal(beacon0,
                generate_carrier_with_random_offset())
        modulated_beacon1 = modulate_signal(beacon1,
                generate_carrier_with_random_offset())
        modulated_beacon2 = modulate_signal(beacon2,
                generate_carrier_with_random_offset())
        modulated_beacon3 = modulate_signal(beacon3,
                generate_carrier_with_random_offset())
        modulated_beacon = [modulate_signal(b,
                generate_carrier_with_random_offset()) for b in beacon]
else:
        modulated_beacon0 = modulate_signal(beacon0,
                generate_carrier())
        modulated_beacon1 = modulate_signal(beacon1,
                generate_carrier())
        modulated_beacon2 = modulate_signal(beacon2,
                generate_carrier())
        modulated_beacon3 = modulate_signal(beacon3,
                generate_carrier())
        modulated_beacon = [modulate_signal(b,
                generate_carrier()) for b in beacon]

def simulate_by_sample_offset(offset):
	offset1 = offset[1]
	offset2 = offset[2]
	offset3 = offset[3]
	shifted0 = modulated_beacon0 * 10
	shifted1 = modulated_beacon1 * 10
	shifted2 = modulated_beacon2 * 10
	shifted3 = modulated_beacon3 * 10
	shifted1 = np.roll(shifted1, offset1)
	shifted2 = np.roll(shifted2, offset2)
	shifted3 = np.roll(shifted3, offset3)
	shifted = [np.roll((modulated_beacon[i] * 10), offset[i]) for i in range(len(offset))]

	superposed = [shifted0[i] + shifted1[i] + shifted2[i]
		+ shifted3[i] for i in range(signal_length)]

	superposed = shifted[0]
	for i in range(1, len(shifted)):
		for j in range(len(shifted[0])):
			superposed[j] += shifted[i][j]
	return superposed

def simulate_by_location(x, y):
	distance = [sqrt((x - sp[0]) ** 2 + (y - sp[1]) ** 2) for sp in speakers]

	t_diff = [(d - distance[0]) / v for d in distance]

	# Convert to the delay / advance in sample
	# positive offset = delay, negative offset = advance

	sample_offset = [int(t * sampling_rate) for t in t_diff]
	return simulate_by_sample_offset(sample_offset)

def add_random_noise(signal, intensity):
	"""
	Add noise to a given signal.
	Intensity: the Noise-Signal Ratio.
	"""
	if intensity == 0:
		return signal
	average = sum(signal[0:100000]) / 100000
	for i in range(len(signal)):
		signal[i] = signal[i] + random() * intensity
	return signal

def get_signal_virtual(**kwargs):
	if 'intensity' not in kwargs:
		intensity = 0
	else:
		intensity = kwargs['intensity']
	if 'x' in kwargs and 'y' in kwargs:
		x = kwargs['x']
		y = kwargs['y']
		return add_random_noise(simulate_by_location(x, y), intensity)
	elif 'offsets' in kwargs:
		offsets = kwargs['offsets']
		return add_random_noise(simulate_by_sample_offset(
			offsets), intensity)
	elif 'offset' in kwargs:
		offsets = kwargs['offset']
		return add_random_noise(simulate_by_sample_offset(
			offsets), intensity)
	else:
		raise Exception("Undefined action. None is returned.")
		return None

def test_correlation(cross_correlation, signal_one, signal_two):
#    result_lib = np.convolve(signal_one, signal_two[::-1])
    result_lib = np.array([np.correlate(signal_one, np.roll(signal_two, k)) for k in range(len(signal_two))])
    result_stu = cross_correlation(signal_one, signal_two)
    return result_lib, result_stu

def arrival_time():
	return 0

def test(cross_correlation, identify_peak, arrival_time, test_num):
    # Virtual Test

    # Utility Functions
    def list_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if lst1[i] != lst2[i]: return False
        return True

    test_cases = {1: "Cross-correlation", 2: "Identify peaks", 3: "Arrival time"}

    # 1. Cross-correlation function
    # If you tested on the cross-correlation section, you should pass this test
    if test_num == 1:
        signal_one = [1, 4, 5, 6, 2]
        signal_two = [1, 2, 0, 1, 2]
        test = list_eq(cross_correlation(signal_one, signal_two), np.convolve(signal_one, signal_two[::-1]))
        if not test:
            print("Test {0} {1} Failed".format(test_num, test_cases[test_num]))
        else: print("Test {0} {1} Passed".format(test_num, test_cases[test_num]))

    # 2. Identify peaks
    if test_num == 2:
        test1 = identify_peak(np.array([1, 2, 2, 199, 23, 1])) == 3
        test2 = identify_peak(np.array([1, 2, 5, 7, 12, 4, 1, 0])) == 4
        your_result1 = identify_peak(np.array([1, 2, 2, 199, 23, 1]))
        your_result2 = identify_peak(np.array([1, 2, 5, 7, 12, 4, 1, 0]))
        if not (test1 and test2):
            print("Test {0} {1} Failed: Your peaks [{2},{3}], Correct peaks [3,4]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your peaks [{2},{3}], Correct peaks [3,4]".format(test_num, test_cases[test_num], your_result1, your_result2))
    # 3. Virtual Signal
    if test_num == 3:
        transmitted = np.roll(beacon[0], -10) + np.roll(beacon[1], -103) + np.roll(beacon[2], -336)
        offsets = arrival_time(beacon[0:3], transmitted)
        test = (offsets[0] - offsets[1]) == (103-10) and (offsets[0] - offsets[2]) == (336-10)
        your_result1 = (offsets[0] - offsets[1])
        your_result2 = (offsets[0] - offsets[2])
        if not test:
            print("Test {0} {1} Failed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))

def plot_speakers(plt, coords, distances, xlim=None, ylim=None):
    """Plots speakers and circles indicating distances on a graph.
    coords: List of x, y tuples
    distances: List of distance from center of circle"""
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    xs, ys = zip(*coords)
    fig = plt.gcf()
    plt.scatter(xs, ys, marker='x', color='b', label='Speakers')
    for i, point in enumerate(coords):
        fig.gca().add_artist(plt.Circle(point, distances[i], facecolor='none',
                                        ec = colors[i]))
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.axis('equal')
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)


def test_identify_offsets(identify_offsets, separate_signal, average_sigs):
	# Utility Functions
	def list_float_eq(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 0.00001: return False
	    return True

	def list_sim(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 3: return False
	    return True

	test_num = 0

	# 1. Identify offsets - 1
	print(" ------------------ ")
	test_num += 1
	test_signal = get_signal_virtual(offsets = [0, 254, 114, 22, 153, 625])
	raw_signal = demodulate_signal(test_signal)
	sig = separate_signal(raw_signal)
	avgs = average_sigs(sig)
	offsets = identify_offsets(avgs)
	test = list_sim(offsets, [0, 254, 114, 23, 153, 625])
	print("Test positive offsets")
	print("Your computed offsets = {}".format(offsets))
	print("Correct offsets = {}".format([0, 254, 114, 23, 153, 625]))
	if not test:
	    print(("Test {0} Failed".format(test_num)))
	else:
	    print("Test {0} Passed".format(test_num))

	# 2. Identify offsets - 2
	print(" ------------------ ")
	test_num += 1
	test_signal = get_signal_virtual(offsets = [0, -254, 0, -21, 153, -625])
	raw_signal = demodulate_signal(test_signal)
	sig = separate_signal(raw_signal)
	avgs = average_sigs(sig)
	offsets = identify_offsets(avgs)
	test = list_sim(offsets, [0, -254, 0, -21, 153, -625])
	print("Test negative offsets")
	print("Your computed offsets = {}".format(offsets))
	print("Correct offsets = {}".format([0, -254, 0, -21, 153, -625]))
	if not test:
	    print("Test {0} Failed".format(test_num))
	else:
	    print("Test {0} Passed".format(test_num))

def test_offsets_to_tdoas(offsets_to_tdoas):
	# 3. Offsets to TDOA

	def list_float_eq(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 0.00001: return False
	    return True

	print(" ------------------ ")
	test_num = 1
	off2t = offsets_to_tdoas([0, -254, 0, -21, 153, -625], 44100)
	test = list_float_eq(np.around(off2t,6), np.around([0.0, -0.005759637188208617, 0.0, -0.0004761904761904762, 0.0034693877551020408, -0.01417233560090703],6))
	print("Test TDOAs")
	print("Your computed TDOAs = {}".format(np.around(off2t,6)))
	print("Correct TDOAs = {}".format(np.around([0.0, -0.005759637188208617, 0.0, -0.0004761904761904762, 0.0034693877551020408, -0.01417233560090703],6)))
	if not test:
	    print("Test Failed")
	else:
	    print("Test Passed")

def test_signal_to_distances(signal_to_distances):
	def list_float_eq(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 0.00001: return False
	    return True
	# 4. Signal to distances
	print(" ------------------ ")
	test_num = 1
	dist = signal_to_distances(demodulate_signal(get_signal_virtual(x=1.765, y=2.683)), 0.009437530220245524)
	test = list_float_eq(np.around(dist,1), np.around([3.2114971586473495, 4.1991869545657172, 2.9105604239534717, 3.9754134851779623, 1.7762604239534723, 2.7870991994636762],1))
	print("Test computed distances")
	print("Your computed distances = {}".format(np.around(dist,1)))
	print("Correct distances = {}".format(np.around([3.2114971586473495, 4.1991869545657172, 2.9105604239534717, 3.9754134851779623, 1.7762604239534723, 2.7870991994636762],1)))
	if not test:
	    print("Test Failed")
	else:
	    print("Test Passed")

# Model the sending of stored beacons, first 2000 samples
sent_0 = beacon[0][:2000]
sent_1 = beacon[1][:2000]
sent_2 = beacon[2][:2000]

# Model our received signal as the sum of each beacon, with some delay on each beacon.
delay_samples0 = 0;
delay_samples1 = 0;
delay_samples2 = 0;
received = np.roll(sent_0,delay_samples0) + np.roll(sent_1,delay_samples1) + np.roll(sent_2,delay_samples2)

def pltBeacons(delay_samples0,delay_samples1,delay_samples2):
    received_new = np.roll(sent_0,delay_samples0) + np.roll(sent_1,delay_samples1) + np.roll(sent_2,delay_samples2)
    plt.figure(figsize=(10,4))
    plt.subplot(2, 1, 1)
    plt.plot(received_new), plt.title('Received Signal (sum of beacons)'), plt.xlabel('Samples'), plt.ylabel('Amplitude')

    ax = plt.subplot(2, 1, 2)
    corr0 = cross_correlation(received_new, sent_0)
    corr1 = cross_correlation(received_new, sent_1)
    corr2 = cross_correlation(received_new, sent_2)
    plt.plot(range(-1000,1000), np.roll(corr0, 1000))
    plt.plot(range(-1000,1000), np.roll(corr1, 1000))
    plt.plot(range(-1000,1000), np.roll(corr2, 1000))
    plt.legend( ('Corr. with Beacon 0', 'Corr. with Beacon 1', 'Corr. with Beacon 2') )
    plt.title('Cross-correlation of received signal and stored copy of Beacon n')
    plt.xlabel('Samples'), plt.ylabel('Correlation')
    plt.tight_layout()
    plt.draw()

def sliderPlots():
    interact(pltBeacons, delay_samples0 = (-500,500,10), delay_samples1 = (-500,500,10), delay_samples2 = (-500,500,10))

def load_corr_sig(identify_peak):
    received_signal = np.load('sig.npy')
    # Plot the received signal
    plt.figure(figsize=(18,4))
    plt.plot(received_signal)
    # Convert the received signals into the format our functions expect
    demod = demodulate_signal(received_signal)
    Lperiod = len(beacon[0])
    Ncycle = len(demod) // Lperiod
    sig = []
    # Iterate through beacons
    for ib, b in enumerate(beacon[:4]):
        s = cross_correlation(demod[0:Lperiod],b)
        # Iterate through cycles
        for i in range(1,Ncycle):
            s = np.hstack([s, cross_correlation(demod[i*Lperiod:(i+1)*Lperiod], b)])
        if ib==0: sig = s
        else:     sig = np.vstack([sig, s])
    sig = [average_signal(s) for s in sig]

    # Plot the cross-correlation with each beacon
    plt.figure(figsize=(18,4))
    for i in range(len(sig)):
        plt.plot(range(len(sig[i])), sig[i], label="Beacon %d"%(i+1))
    plt.legend()

    # Scale the x axis to show +/- 1000 samples around the peaks of the cross correlation
    peak_times = ([identify_peak(sig[i]) for i in range(len(sig))])
    plt.xlim(max(min(peak_times)-1000, 0), max(peak_times)+1000)

def separate_signal(raw_signal):
    """Separate the beacons by computing the cross correlation of the raw signal
    with the known beacon signals.

    Args:
    raw_signal (np.array): raw signal from the microphone composed of multiple beacon signals

    Returns (list): each entry should be the cross-correlation of the signal with one beacon
    """
    Lperiod = len(beacon[0])
    Ncycle = len(raw_signal) // Lperiod
    for ib, b in enumerate(beacon):
        c = cross_correlation(raw_signal[0:Lperiod],b)
        # Iterate through cycles
        for i in range(1,Ncycle):
            c = np.hstack([c, cross_correlation(raw_signal[i*Lperiod:(i+1)*Lperiod], b)])
        if (ib==0): cs = c
        else:       cs = np.vstack([cs, c])
    return cs

def average_sigs(cross_correlations):
    Lperiod = len(beacon[0])
    Ncycle = len(cross_correlations[0]) // Lperiod
    avg_sigs = []
    for c in cross_correlations:
        reshaped = c.reshape((Ncycle,Lperiod))
        averaged = np.mean(np.abs(reshaped),0)
        avg_sigs.append(averaged)

    return avg_sigs

def signal_generator(x, y, noise=0):
    raw_signal = add_random_noise(simulate_by_location(x, y), noise)
    return demodulate_signal(raw_signal)

def plot_average_sigs(beacon_num=0):
    test_signal = np.roll(signal_generator(1.2, 3.4, noise=25), 5000)
    cs = separate_signal(test_signal)
    period_len = len(beacon[0])

    f, axarr = plt.subplots(2, sharex=True, sharey=True,figsize=(17,8))
    axarr[1].set(xlabel='Sample Number')
    period_ticks = np.arange(0, len(cs[0]), period_len)
    axarr[1].xaxis.set_ticks(period_ticks)

    axarr[0].plot(np.abs(cs[beacon_num]))
    [axarr[0].axvline(x=line, color = "red", linestyle='--') for line in period_ticks]
    axarr[0].set_title('2.5 sec Recording of Beacon 1 After Separation\n(No Averaging)')

    avgs = average_sigs(cs)
    axarr[1].plot(avgs[beacon_num])
    axarr[1].axvline(x=period_ticks[0], color = "red", linestyle='--', label='period start')
    axarr[1].axvline(x=period_ticks[1], color = "red", linestyle='--')
    axarr[1].set_title('Averaged & Centered Periodic Output for Beacon 1')
    plt.legend()

    stacked_cs = np.abs(cs[beacon_num])[:(len(cs[beacon_num])//period_len)*period_len].reshape(-1, period_len)
    print("Samples Offset of Each Period in Non-Averaged Signal:",[np.argmax(s) for s in stacked_cs])
    print("Samples Offset in Averaged Signal:",[np.argmax(avgs[beacon_num])])

def plot_shifted(identify_peak):
    # Simulate the received signal
    test_signal = signal_generator(1.4, 3.22)

    # Separate the beacon signals by demodulating the received signal
    separated = separate_signal(test_signal)

    # Perform our averaging function
    avgs = average_sigs(separated)
    # Plot the averaged output for each beacon
    plt.figure(figsize=(16,4))
    for i in range(len(avgs)):
        plt.plot(avgs[i], label="{0}".format(i))
    plt.title("Separated and Averaged Cross-correlation outputs with Beacon0 at t=0")
    plt.legend()
    plt.show()

    # Plot the averaged output for each beacon centered about beacon0
    plt.figure(figsize=(16,4))
    peak0 = identify_peak(avgs[0])
    Lperiod = len(avgs[0])
    for i in range(len(avgs)):
        plt.plot(np.roll(avgs[i], Lperiod//2 - peak0), label="{0}".format(i))
    plt.title("Shifted Cross-correlated outputs centered about Beacon0")
    plt.legend()
    plt.show()

################################################
# for infinite periodic cross correlation plot #
################################################

def correlation_plots(offset):
    stationary_coord = np.arange(-10,11)
    stationary_sig = np.array([-1, 0, 1, 0] * 5 + [-1])
    sliding_sig = np.array([-0.5, 0, 0.5, 0, -0.5])
    sliding_coord = np.array([-2,-1,0,1,2])
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(15,5))

    # plot stationary and sliding signal
    ax1.set_xlim(-10,10)
    ax1.plot(stationary_coord, stationary_sig, label = "infinite periodic stationary signal", marker='o')
    ax1.plot(sliding_coord+offset, sliding_sig, color="orange", label = "sliding signal", marker='o')
    ax1.plot(np.arange(-10-8,-1)+offset, [0]*17, color="orange")
    ax1.plot(np.arange(2,11+8)+offset, [0]*17, color="orange")
    ax1.axvline(offset, color = "black", ls="--")
    ax1.set_xticks(np.arange(-10, 11, 1.0))
    ax1.set_ylim(-1.2, 1.2)
    ax1.legend()

    # plot corr result
    corr = np.correlate(stationary_sig, sliding_sig, "full")[12-2:12+3]
    x = np.arange(-2,3,1)
    ax2.set_xlim(-10,10)
    ax2.set_ylim(-2, 2)
    ax2.plot(x, corr, label="infinitely periodic cross correlation", color="g")
    index = (offset+2)%4 - 2
    ax2.scatter(index, corr[index+2], color = "r")
    ax2.axvline(index, color = "black", ls="--")
    ax2.set_xticks(np.arange(-10, 11, 1.0))
    ax2.legend()
    ax2.set_title("cross_correlation([-1, 0, 1, 0, -1], [-0.5, 0, 0.5, 0, -0.5])")

    ax1.set_title("Infinite Periodic Linear Cross Correlation\nCorr Val at offset "+str(offset)+" is "+str(corr[index+2]))
    plt.show()

def inf_periodic_cross_corr():
    # interactive widget for playing with the offset and seeing the cross-correlation peak and aligned signals.
    widget = interactive(correlation_plots, offset=(-8, 8, 1))
    display(widget)
