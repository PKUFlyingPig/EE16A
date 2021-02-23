import argparse
import numpy as np
import sys
from PIL import Image
from PIL import ImageQt
import time

from PyQt5 import QtGui, QtCore, QtWidgets


# Screen dimensions
DEFAULT_DISP_WIDTH = 720
DEFAULT_DISP_HEIGHT = 720


help_menu = """Quit: press [Esc], [Ctrl+Q], or [Ctrl+W] at any time to exit\n
Help: press [H] to show this help menu\n"""



class Mask(QtWidgets.QWidget):
  def __init__(self, imgWidth, imgHeight, infile, imagefile, sleepTime, noise):
    super(Mask, self).__init__()

    self.setAutoFillBackground(True)
    p = self.palette()
    p.setColor(self.backgroundRole(), QtCore.Qt.black)
    self.setPalette(p)

    # Set up shortcuts to close the program
    QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.close)
    QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+W"), self, self.close)
    QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.close)
    QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self.close)
    QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, self.close)
    QtWidgets.QShortcut(QtGui.QKeySequence("H"), self, self.help)

    self.imgWidth = imgWidth        # Matrix width
    self.imgHeight = imgHeight      # Matrix height
    
    self.mask_file = infile         # Mask File

    self.image = np.array(Image.open(imagefile).convert(mode = 'L'))        # Load image as np array

    # Load the imaging mask
    self.Hr = np.load(self.mask_file)        # Masking matrix

    # Measurement rows
    self.numMeasurements = self.Hr.shape[0]


    self.dispWidth = DEFAULT_DISP_WIDTH      # Display width
    self.dispHeight = DEFAULT_DISP_HEIGHT    # Display height

    self.sleepTime = sleepTime          # Gap between consecutive measurements

    self.count = 0
    self.fullscreen = True
    self.sensor_readings = np.zeros((self.imgHeight * self.imgWidth, 1))
    self.noise = noise
    self.time0 = time.time()                          # Will be used to time scan
    self.time_final = 0

    # Set window size and center
    self.resize(self.dispWidth, self.dispHeight)
    self.center()

    # Create a label (used to display the image)
    self.label = QtWidgets.QLabel(self)
    self.label.setAlignment(QtCore.Qt.AlignCenter)

    self.col = QtGui.QColor(0, 0, 0)
    self.label.setGeometry(QtCore.QRect(0, 0, self.dispWidth, self.dispHeight))

    # Set up the timer
    self.timer = QtCore.QTimer()
    self.showNormal()
    self.started=True
    self.timer.start(sleepTime)
    self.timer.timeout.connect(self.updateData)

  def center(self):
    qr = self.frameGeometry()
    cp = QtWidgets.QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def help(self):
    self.fullscreen = True
    self.showNormal()
    print(help_menu)

  def updateData(self):

    if self.count >= self.numMeasurements:
      """
      # Vectorize and save sensor readings
      self.sensor_readings = np.reshape(self.sensor_readings, (self.numMeasurements, 1))
      print("Saving data as %s.npy" % self.outfile)
      np.save("%s.npy" % self.outfile, self.sensor_readings)
      """

      self.timer.stop()
      self.time_final = time.time()
      elapsed_time = self.time_final - self.time0
      print("\nScan completed")
      print("Scan time: %.3f m" % ((elapsed_time//60)), " %.3f s" % ((elapsed_time)%60))
      time.sleep(2.0)
      self.close()

    else:
      mask = self.Hr[self.count, :]                   # Load a mask as a row vector
      img_vec = np.reshape(self.image, (self.imgWidth * self.imgHeight, 1))      # Image vector
    
      curr_scan = np.dot(mask, img_vec)      # Perform a scan using dot product of mask row vec and image col vec
      self.sensor_readings[self.count] = curr_scan               # Add to vector of sensor reading
      
      if self.noise:
        scan_noise = np.random.normal(0, self.noise)
        self.sensor_readings[self.count] += scan_noise

      curr_res = np.reshape(self.sensor_readings, (self.imgHeight, self.imgWidth))

      # Conversion to QtImage
      #curr_res = (curr_res * 255).astype(np.uint8)
      curr_res = (curr_res).astype(np.uint8)
      res = Image.fromarray(curr_res, mode = 'L')
      resQT = ImageQt.ImageQt(res) 
      QI = resQT
      self.label.setPixmap(QtGui.QPixmap.fromImage(QI).scaled(self.dispWidth, self.dispHeight))

      if self.count <= 10 or self.count % 100 == 0:
        print("Count: %s \n" % self.count)

      self.count += 1  


def main():
  print("\nEECS16A Imaging Lab\n")

  # Parse arguments
  parser = argparse.ArgumentParser(description = 'This program projects a sampling pattern and records the corresponding phototransistor voltage.')
  parser.add_argument('--width', type = int, default = 32, help = 'width of the image in pixels (default = 32px)')
  parser.add_argument('--height', type = int, default = 32, help = 'height of the image in pixels (default = 32px)')
  parser.add_argument('--mask', default = "../saved_data/H.npy", help = 'saved sampling pattern mask (default = "../saved_data/H.npy")')
  parser.add_argument('--image', default = "../saved_data/Home.png", help = 'saved image (default = "../saved_data/Home.png")')
  #parser.add_argument('--out', default = "../saved_data/sensor_readings_H", help = 'output path (default="../saved_data/sensor_readings_H", saved as "sensor_readings_H.npy")')
  parser.add_argument('--sleepTime', type = int, default = 100, help = 'sleep time in milliseconds -- time between projector update and data capture')
  parser.add_argument('--noise', type = int, default = 0, help = 'noise parameter to simulate non-ideal imaging')

  args = parser.parse_args()

  print("Sleep time in ms: %d" % args.sleepTime)

  print("Image width: %d" % args.width)
  print("Image height: %d" % args.height)

  print("Mask file: %s \n" % args.mask)
  print("Image file: %s \n" % args.image)

  #print("Output file: %s \n" % args.out)

  app = QtWidgets.QApplication(sys.argv)
  mask = Mask(args.width, args.height, args.mask, args.image, args.sleepTime, args.noise)

  sys.exit(app.exec_())

if __name__ == '__main__':
  main()

