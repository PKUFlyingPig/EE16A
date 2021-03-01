from __future__ import division
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from IPython.html.widgets import *
#from ipywidgets import *
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt
from random import random
import scipy.io
import scipy.io.wavfile
import scipy.signal

# Constants
v = 340.29
v_air = 340.29
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

def cross_correlation(signal1, signal2):
    """Compute the cross_correlation of two given periodic signals
    Args:
    signal1 (np.array): input signal 1
    signal2 (np.array): input signal 2

    Returns:
    cross_correlation (np.array): cross-correlation of signal1 and signal2

    >>> cross_correlation([0, 1, 2, 3], [0, 2, 3, 0])
    [8, 13, 6, 3]
    """
    return np.fft.ifft(np.fft.fft(signal1) * np.fft.fft(signal2).conj()).real

def cross_corr_demo_1():
    """Compute the cross_correlation of two given periodic signals
    Args:
    signal1 (np.array): input signal 1
    signal2 (np.array): input signal 2

    Returns:
    cross_correlation (np.array): circular cross-correlation of signal1 and signal2

    >>> cross_correlation([0, 1, 2, 3], [0, 2, 3, 0])
    [8, 13, 6, 3]
    """
    # Input signals for which to compute the cross-correlation
    signal1 = np.array([1, 2, 3, 2, 1, 0])
    signal2 = np.array([3, 1, 0, 0, 0, 1])
    print('input signal1: '+str(signal1))
    print('input signal2: '+str(signal2))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal1, np.roll(signal2,k))[0] for k in range(len(signal2))]
    print( 'cross-correlation:'+str(corr))

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(12,6))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal1, 'bo-', label='Signal 1')
        plt.plot(signal2, 'rx-', label='Signal 2')
        plt.xlim(0, 5)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        plt.title('Computed cross-correlation(%d)=%d\n%s\n%s'%(i, np.dot(signal1, signal2), str(signal1), str(signal2)))
        signal2 = np.roll(signal2, 1)

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
    print('input signal1: '+str(signal1))
    print('input signal2: '+str(signal2))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal1, np.roll(signal2,k))[0] for k in range(len(signal2))]
    print( 'cross-correlation:'+str(corr))

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(12,12))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal1, 'bo-', label='Signal 1')
        plt.plot(signal2, 'rx-', label='Signal 2')
        plt.xlim(0, 11)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        plt.title('Computed cross-correlation(%d)=%d\n%s\n%s'%(i, np.dot(signal1, signal2), str(signal1), str(signal2)))
        signal2 = np.roll(signal2, 1)

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
        transmitted = np.roll(beacon[0], 10) + np.roll(beacon[1], 103) + np.roll(beacon[2], 336)
        offsets = arrival_time(beacon[0:3], transmitted)
        test = (offsets[0] - offsets[1]) == (103-10) and (offsets[0] - offsets[2]) == (336-10)
        your_result1 = (offsets[0] - offsets[1])
        your_result2 = (offsets[0] - offsets[2])
        if not test:
            print("Test {0} {1} Failed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))

def plot_speakers(plt, coords, distances, xlim=None, ylim=None, circle=True):
    """Plots speakers and circles indicating distances on a graph.
    coords: List of x, y tuples
    distances: List of distance from center of circle"""
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    xs, ys = zip(*coords)
    fig = plt.gcf()

    for i in range(len(xs)):
    	# plt.scatter(xs[i], ys[i], marker='x', color=colors[i], label='Speakers')
    	plt.scatter(xs[i], ys[i], marker='x', color=colors[i])


    if circle==True:
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

def plot_average_sigs():
    test_signal = np.roll(signal_generator(1.2, 3.4, noise=25), 5000)
    cs = separate_signal(test_signal)

    plt.figure(figsize=(16,4))
    plt.plot(np.abs(cs[0]))
    plt.title('2.5 sec Recording of Beacon 1 After Separation\n(No Averaging)')
    plt.xlabel('Sample Number')
    plt.show()

    avgs = average_sigs(cs)
    plt.figure(figsize=(5,4))
    plt.plot(avgs[0])
    plt.title('Averaged & Centered Periodic Output for Beacon 1')
    plt.xlabel('Sample Number')
    plt.show()

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

##############
# FROM APS 1 #
##############
def identify_peak(signal):
    """Returns the index of the peak of the given signal."""
    peak_index = np.argmax(signal)
    return peak_index


"""
def arrival_time(beacon, signal):
    arrival_time = [None]*len(beacon);
    for i in range(len(beacon)):
        sent_temp = beacon[i][:1000]
        ccorr_temp = cross_correlation(sent_temp,signal);
        arrival_time[i] = identify_peak(ccorr_temp);
    return arrival_time
    """

def arrival_time(beacons, signal):
    """Returns a list of arrival times (in samples) of each beacon signal.
    Args:
    beacons (list): list in which each element is a numpy array representing one of the beacon signals
    signal (np.array): input signal, for example the values recorded by the microphone

    Returns:
    arrival_time [samples] (list): arrival time of the beacons in the order that they appear in the input
    (e.g. [arrival of beacons[0], arrival of beacons[1]...])
    """
    # YOUR CODE HERE
    return [identify_peak(cross_correlation(signal, b)) for b in beacons]

def identify_offsets(averaged):
    """ Identify peaks in samples.
    The peaks of the signals are shifted to the center."""

    # Reshaping (shifting) the input so that all of our peaks are centered about the peak of beacon0
    peak = identify_peak(averaged[0])
    shifted = [np.roll(avg, len(averaged[0]) // 2 - peak) for avg in averaged]

    peaks = [identify_peak(shifted[i]) for i in range(len(shifted))]
    peak_offsets = peaks - peaks[0]

    return peak_offsets


def offsets_to_tdoas(offsets, sampling_rate):
    """ Convert a list of offsets to a list of TDOA's """
    TDOA = list(np.array(offsets) / sampling_rate)
    return TDOA


def signal_to_offsets(raw_signal):
    """ Compute a list of offsets from the microphone to each speaker. """

    separated = separate_signal(raw_signal)
    averaged = average_sigs(separated)
    offsets = identify_offsets(averaged)

    return offsets


def signal_to_distances(raw_signal, t0):
    """ Returns a list of distancs from the microphone to each speaker. """

    offsets = signal_to_offsets(raw_signal)
    tdoas = offsets_to_tdoas(offsets, sampling_rate)
    all_dists = list(v_air*(t0 + np.array(tdoas)))

    return all_dists


def draw_hyperbola(p1, p2, d):
    """ hyperbola drawing function """

    p1=np.matrix(p1)
    p2=np.matrix(p2)
    pc=0.5*(p1+p2)
    p21=p2-p1
    d21=np.linalg.norm(p21)
    th=np.array(np.matrix(list(range(-49,50)))*pi/100) #radian sweep list
    a=d/2
    b=((d21/2)**2-(a)**2)**0.5
    x=a/np.cos(th)
    y=b*np.tan(th) #hyperbola can be represented as (d*sec(th),d*tan(th))
    p=np.vstack((x,y))
    m=np.matrix([[p21[0,0]/d21, -p21[0,1]/d21],[p21[0,1]/d21, p21[0,0]/d21]]) #rotation matrix
    vc=np.vstack((pc[0,0]*np.ones(99),pc[0,1]*np.ones(99))) #offset
    return np.transpose(m*p+vc)


def get_signal_from_wav(wavFileName):
    """Get the signal from a pre-recorded wav file"""
    return mic.new_data_wav(wavFileName)

#########
# APS 2 #
#########

def hyperbola_demo_1():
    # Assume we already know the time of arrival of the first beacon, which is R0/(speed_of_sound)
    coords = [(0, 0), (5, 0), (0, 5), (5, 5)]
    coord_mic = (1.2, 3.6) #microphone position
    received_signal = get_signal_virtual(x=coord_mic[0], y=coord_mic[1])
    demod = demodulate_signal(received_signal)
    distances = signal_to_distances(demod, ((coord_mic[0])**2+(coord_mic[1])**2)**0.5/v_air)
    distances = distances[:4]
    print("The distances are: " + str(distances))
    TDOA = offsets_to_tdoas(signal_to_offsets(demod), sampling_rate)
    plt.figure(figsize=(8,8))
    dist=np.multiply(v,TDOA)
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    for i in range(3):
        hyp=draw_hyperbola(coords[i+1], coords[0], dist[i+1]) #Draw hyperbola
        plt.plot(hyp[:,0], hyp[:,1], color=colors[i+1], label='Hyperbola for beacon '+str(i+1), linestyle='--')
    plot_speakers(plt, coords, distances)
    plt.xlim(-9, 18)
    plt.ylim(-6, 6)
    plt.legend()
    plt.show()

def plot_speakers_demo():
    # Plot the speakers
    plt.figure(figsize=(8,8))
    coords = [(0, 0), (5, 0), (0, 5), (5, 5)]
    coord_mic = (1.2, 3.6) #microphone position
    received_signal = get_signal_virtual(x=coord_mic[0], y=coord_mic[1])
    demod = demodulate_signal(received_signal)
    distances = signal_to_distances(demod, ((coord_mic[0])**2+(coord_mic[1])**2)**0.5/v_air)
    distances = distances[:4]

    # Plot the linear relationship of the microphone and speakers.
    isac=1; #index of the beacon to be sacrificed
    TDOA = offsets_to_tdoas(signal_to_offsets(demod), sampling_rate)
    helper = lambda i: float(speakers[i][0]**2+speakers[i][1]**2)/(v*TDOA[i])-float(speakers[isac][0]**2+speakers[isac][1]**2)/(v*TDOA[isac])
    helperx = lambda i: float(speakers[i][0]*2)/(v*TDOA[i])-float(speakers[isac][0]*2)/(v*TDOA[isac])
    helpery = lambda i: float(speakers[i][1]*2)/(v*TDOA[i])-float(speakers[isac][1]*2)/(v*TDOA[isac])

    x = np.linspace(-9, 9, 1000)
    y1,y2,y3 = [],[],[]
    if isac!=1: y1 = [((helper(1)-helper(isac))-v*(TDOA[1]-TDOA[isac])-helperx(1)*xi)/helpery(1) for xi in x]
    if isac!=2: y2 = [((helper(2)-helper(isac))-v*(TDOA[2]-TDOA[isac])-helperx(2)*xi)/helpery(2) for xi in x]
    if isac!=3: y3 = [((helper(3)-helper(isac))-v*(TDOA[3]-TDOA[isac])-helperx(3)*xi)/helpery(3) for xi in x]

    # You can calculate and plot the equations for the other 2 speakers here.
    if isac!=1: plt.plot(x, y1, label='Equation for beacon 1', color='g')
    if isac!=2: plt.plot(x, y2, label='Equation for beacon 2', color='c')
    if isac!=3: plt.plot(x, y3, label='Equation for beacon 3', color='y')
    plt.legend()
    plot_speakers(plt, coords, distances)
    plt.xlim(-9, 11)
    plt.ylim(-6, 6)
    plt.show()

def construct_system_test(construct_system):
    coords = [(0, 0), (5, 0), (0, 5), (5, 5)]
    coord_mic = (1.2, 3.6) #microphone position
    received_signal = get_signal_virtual(x=coord_mic[0], y=coord_mic[1])
    demod = demodulate_signal(received_signal)
    distances = signal_to_distances(demod, ((coord_mic[0])**2+(coord_mic[1])**2)**0.5/v_air)
    distances = distances[:4]

    # Plot the linear relationship of the microphone and speakers.
    isac=1; #index of the beacon to be sacrificed
    TDOA = offsets_to_tdoas(signal_to_offsets(demod), sampling_rate)
    A, b = construct_system(speakers,TDOA)
    for i in range(len(b)):
        print ("Row %d: %.f should equal %.f"%(i, A[i][0] * 1.2 + A[i][1] * 3.6, b[i]))

def least_squares_test(least_squares):
    A = np.array(((1,1),(1,2),(1,3),(1,4)))
    b = np.array((6, 5, 7, 10))
    yourres = least_squares(A,b)
    print('Your results: ',yourres)
    correctres = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    print('Correct results: ',correctres)

# Define a helper function to use least squares to calculate position from just the TDOAs
def calculate_position(least_squares, construct_system, speakers, TDOA, isac=1):
    return least_squares(*construct_system(speakers, TDOA, isac))

# Define a testing function
def test_loc(least_squares, construct_system, x_pos, y_pos, inten, debug=False):
    raw_signal = get_signal_virtual(mode="Location", x=x_pos, y=y_pos, intensity=inten)

    # Demodulate raw signal
    demod = demodulate_signal(raw_signal)

    # Separate the beacon signals
    separated = separate_signal(demod)

    # Perform our averaging function
    avg = average_sigs(separated)

    # Calculate offsets and TDOAs
    offsets = identify_offsets(avg)
    TDOA = offsets_to_tdoas(signal_to_offsets(demod), sampling_rate)

    # Construct system of equations
    A, b = construct_system(speakers, TDOA)

    # Calculate least squares solution
    pos = calculate_position(least_squares, construct_system, speakers, TDOA)

    if debug:
        # Plot the averaged output for each beacon
        plt.figure(figsize=(12,6))
        for i in range(len(avg)):
            plt.subplot(3,2,i+1)
            plt.plot(avg[i])
            plt.title("Beacon %d"%i)
        plt.tight_layout()

        # Plot the averaged output for each beacon centered about beacon0
        plt.figure(figsize=(16,4))
        peak = identify_peak(avg[0])
        for i in range(len(avg)):
            plt.plot(np.roll(avg[i], len(avg[0]) // 2 - peak), label="{0}".format(i))
        plt.title("Beacons Detected")
        plt.legend()
        plt.show()

        print( "Offsets (samples): %s"%str(offsets))
        print( "Times (s): [%s]\n"%", ".join(["%0.6f" % t for t in TDOA]))
        print( "Constructing system...")
        print( "Verifying system using known position...")
        for i in range(len(b)):
            print( "Row %d: %.f should equal %.f"%(i, A[i][0] * x_pos + A[i][1] * y_pos, b[i]))

        print( "\nCalculating least squares estimate...")
    print("Expected: (%.3f, %.3f); got (%.3f, %.3f)\n"%(x_pos, y_pos, pos[0], pos[1]))

def simulation_testing(construct_system, least_squares, filename, sim=True, isac=1):
    # LOAD IN SIMULATION DATA
    record_rate, raw_signal = scipy.io.wavfile.read(filename)
    # Get single channel
    if (len(raw_signal.shape) == 2):
        raw_signal = raw_signal[:,0]
    plt.figure(figsize=(16,4))
    plt.title("Raw Imported Signal")
    plt.plot(raw_signal)

    # Demodulate raw signal
    demod = demodulate_signal(raw_signal)
    # Separate the beacon signals
    separated = separate_signal(demod)
    # Perform our averaging function
    avg = average_sigs(separated)
    # Plot the averaged output for each beacon
    fig = plt.figure(figsize=(12,6))
    for i in range(len(avg)):
        plt.subplot(3,2,i+1)
        plt.plot(avg[i])
        plt.title("Extracted Beacon %d"%i)
    plt.tight_layout()

    # Load Beaker Locations for Simulation
    simulation = np.array([(0.,0.),(0.53,0.03),(0.66,0.31),(0.50,0.6),(-0.04,0.58),(-0.15,0.30)])

    # Calculate distances and plot
    distances = signal_to_distances(demod, 0)
    # Calculate quantities and compute least squares solution
    offsets = signal_to_offsets(demod)

    distances = signal_to_distances(demod, 0)
    TDOA = offsets_to_tdoas(offsets, sampling_rate)
    x, y = calculate_position(least_squares, construct_system, simulation, TDOA, isac)

    #print( "Distance differences (m)): [%s]\n"%", ".join(["%0.4f" % d for d in distances]))
    print( "Least Squares Position: %0.4f, %0.4f" % (x, y))
    if (sim): print( "Actual Simulation Microphone Position: (0, 0.4)")

    # Find distance from speaker 0 for plotting
    dist_from_origin = np.linalg.norm([x, y])
    dist_from_speakers = [d + dist_from_origin for d in distances]
    print( "Distances from Speakers : [%s]\n"%", ".join(["%0.4f" % d for d in dist_from_speakers]))

    # Plot speakers and Microphone
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, marker='o', color='r', label='Microphone')
    plot_speakers(plt, simulation, [d for d in dist_from_speakers], circle=False)
    xmin = min(simulation[:,0])
    xmax = max(simulation[:,0])
    xrange = xmax-xmin
    ymin = min(simulation[:,1])
    ymax = max(simulation[:,1])
    yrange = ymax-ymin

    # Plot linear equations for LS
    A, b = construct_system(simulation, TDOA, isac) #for debugging
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    x2 = np.linspace(xmin-xrange*.2, xmax+xrange*.2, 1000)
    j=0;

    for i in range(len(b)):
        if i==isac-1: j=j+2
        else: j=j+1
        y2 = [(b[i] - A[i][0]*xi) / A[i][1] for xi in x2]
        plt.plot(x2, y2, color=colors[j], label="Linear Equation " + str(j), linestyle='-')
        plt.xlim(xmin-xrange*.2, xmax+xrange*.2)
        plt.ylim(ymin-yrange*.2, ymax+yrange*.2)
        plt.legend(bbox_to_anchor=(1.4, 1))

    for i in range(5):
        hyp=draw_hyperbola(simulation[i+1], simulation[0], distances[i+1]) #Draw hyperbola
        plt.plot(hyp[:,0], hyp[:,1], color=colors[i+1], label='Hyperbolic Equation '+str(i+1), linestyle=':')

    plt.xlim(xmin-xrange*.2, xmax+xrange*.2)
    plt.ylim(ymin-yrange*.2, ymax+yrange*.2)
    plt.legend(bbox_to_anchor=(1.6, 1))
    plt.show()

def user_test(construct_system, least_squares, isac=1):
    filename = input("Type filename (including the .wav): ")
    simulation_testing(construct_system, least_squares, filename, sim=False, isac=isac)
