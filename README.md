# LNHR DAC IIa QCoDeS Driver

This repository contains the QCoDeS driver for the Basel Precision Instruments Low Noise High Resolution Digital to Analog Converter IIa or LNHR DAC IIa. Additionally there are some examples on how to use the driver.

## Features

The driver gives you full control of the LNHR DAC IIa through QCoDeS:

- Voltage, bandwidth and enable of each channel.
- Six Arbitrary Waveform Generators (AWG), configurable either by manually setting points or by using the Standard Waveform Generator (SWG) for fast creation of simple waveforms (sine, triangle, pulse, white noise and more). 
- Fast adaptive 2D-scan, with update rates as fast as 10 &mu;s per point, configurable through parameters. The video below shows an adaptive fast 2D-scan with 50000 points at live speed, done by the LNHR DAC IIa. 

*With the LNHR DAC IIa, four fast adaptive 2D-scans can run simultaneously.*

https://github.com/user-attachments/assets/de2fac24-d2c0-4c25-9fb7-104f3bedb5a8

## Setup

Download `Baspi_Lnhrdac2a.py` and `Baspi_Lnhrdac2a_Controller.py` and copy them to your project folder. `qcodes_examples.ipynb` gives some examples on how the driver can be used.

## Further Documentation

See https://www.baspi.ch/manuals for more information on the LNHR DAC IIa.

See https://microsoft.github.io/Qcodes/ for more information about the QCoDeS framework.

If you have purchased an LNHR DAC IIa, you have received a USB stick which includes the full documentation of the LNHR DAC IIa. Please be aware that the official documentation of the LNHR DAC IIa does not include any specific information on how to use the DAC with the QCoDeS framework. However, since the QCoDeS driver of the LNHR DAC IIa allows for full control of the device and is mainly an interface, the general documentation on the LNHR DAC IIa is still useful. The general documentation includes documentation about all commands available to the LNHR DAC IIa.

## Contributing

If you found a bug or are having a serious issue, please use the GitHub issue tracker to report it.