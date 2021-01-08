#!/usr/bin/env python

import time
import logging
import sys
import glob
import serial
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque

# Based on tutorial
# http://josephmr.com/realtime-arduino-sensor-monitoring-with-matplotlib/

# Python logging guidelines
# http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python

MAX_X = 100   # Width of graph
MAX_Y = 3.3  # Height of
MAX_V = 3.3 # Max voltage
line = deque([0.0]*MAX_X, maxlen=MAX_X)

# Serial functions
def serial_ports():
    """Lists serial ports

    Raises:
    EnvironmentError:
        On unsupported or unknown platforms
    Returns:
        A list of available serial ports
    """
    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(256)]

    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this is to exclude your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')

    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')

    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def open_port(port, baud):
    """Open a serial port.

    Args:
    port (string): port to open, on Unix typically begin with '/dev/tty', on
        or 'COM#' on Windows.
    baud (int, optional): baud rate for serial communication

    Raises:
    SerialException: raised for errors from serial communication

    Returns:
       A pySerial object to communicate with the serial port.
    """
    ser = serial.Serial()
    try:
        ser = serial.Serial(port, baud, timeout=10)
        logger.info("Opened serial connection on port %s"%port)
        return ser
    except serial.SerialException:
        raise

# Plotting functions
def update(fn, l2d, ser):
    try:
        data = ser.readline().decode()
        logger.debug("Raw ADC value: %s"%data)

        # Add new point to deque
        line.append(float(data)*MAX_V/4095)

        # Set the l2d to the new line coord ([x-coords], [y-coords])
        l2d.set_data(range(int(-MAX_X/2), int(MAX_X/2)), line)
        #l2d.set_data(range(MAX_X), line)

    except (serial.SerialException, serial.SerialTimeoutException) as e:
        logger.info("Device disconnected")
        quit()
    except Exception as e:
        logger.error('', exc_info=True)
        quit()

def main(device="/dev/ttyACM0", baud=9600, plot=True):
    ser = serial.Serial()
    try:
        ser = open_port(device,baud)
    except:
        logger.error("Cannot open port %s"%device)
        available = serial_ports()
        if available:
            logger.info("The following ports are available: %s\n"%" "\
                    .join(available))
        else:
            logger.info("No serial ports detected.")
        quit()

    # Plot the values in real time
    if args.plot:
        fig = plt.figure()

        # make the axes revolve around [0,0] at the center
        # instead of the x-axis being 0 - +100, make it -50 - +50 ditto
        # for y-axis -512 - +512
        a = plt.axes(xlim=(-(MAX_X/2),MAX_X/2),ylim=(0,MAX_Y*1.15))

        # plot an empty line and keep a reference to the line2d instance
        l1, = a.plot([], [])
        ani = anim.FuncAnimation(fig, update, fargs=(l1,ser), interval=0.001)
        plt.title("Sensor Voltage")
        plt.xlabel("V")
        plt.xlabel("t")
        plt.show()

    else:
        try:
            while True:
                logger.debug(ser.readline().decode())
        except serial.SerialTimeoutException:
            logger.info("Device disconnected")
            quit()
        except serial.SerialException:
            raise
        except Exception as e:
            logger.error('', exc_info=True)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Record photo cell values.')
    parser.add_argument('-D', '--device', choices=serial_ports(), default="/dev/ttyACM0",
            help='a serial port to read from')
    parser.add_argument('-b', '--baud', type=int, default=9600,
            help="baud rate for the serial connection ")
    parser.add_argument('-p', '--plot', action='store_true', default=True,
            help='show real time plot (requires X11 to be enabled)')

    parser.add_argument('--debug', action='store_true', default=False,
            help='print all raw ADC values')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args.device, args.baud, args.plot)
