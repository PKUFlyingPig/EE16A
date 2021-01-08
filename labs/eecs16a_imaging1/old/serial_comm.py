import subprocess
from virtual_oscope import serial_ports

# function for getting the list of ports other than COM1
def get_portlist():
    ports = serial_ports()

    result = []
    if len(ports) > 1:
        for p in ports:
            if p != "COM1":
                result.append(p)    
    else:    
        print("No serial ports other than COM1 detected. Make sure MSP430 is connected and try again.")
    return result

def main():
    portlist = get_portlist()

    # list the ports seen on the machine
    print("You have the following ports:")
    counter = 1
    for p in portlist:
        print("%i - %s"%(counter,p))
        counter = counter + 1

    # let the user choose the correct port
    index = int(input("Select the index of the port corresponding to MSP Application UART1 in device manager: \n"))
    while index > len(portlist):
        print("You have only %i COM ports to choose from, please enter an index in [1, %i]"%(len(portlist),len(portlist)))
        index = int(input("Select the index of the port corresponding to MSP Application UART1 in device manager: \n"))

    # connect to the port
    subprocess.call(["python", "virtual_oscope.py", "-D", portlist[index-1], "--debug"])

if __name__ == '__main__':
    main()