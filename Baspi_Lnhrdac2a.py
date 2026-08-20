# ----------------------------------------------------------------------------------------------------------------------------------------------
# LNHR DAC IIa QCoDeS driver
# v0.3.0 
# Copyright (c) Basel Precision Instruments AG (2026)
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be 
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  
# If not, see <https://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------------------------------------------------------------------------

# imports --------------------------------------------------------------

from Baspi_Lnhrdac2a_Controller import BaspiLnhrdac2aController

from qcodes.station import Station
from qcodes.instrument import VisaInstrument, InstrumentChannel, ChannelList, InstrumentModule
from qcodes.parameters import ParameterWithSetpoints, create_on_off_val_mapping
import qcodes.validators as validate

from numpy import ndarray, array, linspace, zeros, eye, ones, empty, tile, full, concatenate
from numpy.linalg import inv, det
from functools import partial
from dataclasses import dataclass
from time import sleep, perf_counter

# logging --------------------------------------------------------------

import logging

log = logging.getLogger(__name__)

# class ----------------------------------------------------------------

class BaspiLnhrdac2aLockingValidator(validate.Validator):

    def __init__(self, submodule: any):
        """
        This class implements a validator that can be used to lock any submodule of the main instrument.
        The validator checks the locked-attribute inside the submodule. If True, the validator raises an error.

        Parameters:
        submodule: reference of submodule the locked-attribute is a part of

        Raises:
        ValueError: the submodule this parameter is a part of is locked
        """
        self.submodule = submodule

    def validate(self, value: any, context = "BaspiLnhrdac2LockingValidator") -> None:
        """
        Validates if the locked-attribute is False. 
        """

        if self.submodule.locked:
            raise ValueError(f"Submodule {self.submodule} has been locked and is currently not accessible.")
        

# class ----------------------------------------------------------------

class BaspiLnhrdac2aChannel(InstrumentChannel):

    def __init__(self, 
                 parent: VisaInstrument, 
                 name: str, 
                 channel: int, 
                 controller: BaspiLnhrdac2aController):
        """
        Class that defines a channel of the LNHR DAC II with all its QCoDeS-parameters.

        Channel-Parameters:
        voltage (-10.0 V ... +10.0 V)
        high_bandwidth (ON/True: 100 kHz, OFF/False: 100 Hz)
        enable (ON/True: channel on, OFF/False: channel off)

        Parameters:
        parent: instrument this channel is a part of
        name: name of the channel
        channel: channel numnber
        controller: the controller the instrument uses for its communication
        """

        super().__init__(parent, name)

        self.voltage = self.add_parameter(
            name = "voltage",
            unit = "V",
            get_cmd = partial(controller.get_channel_dacvalue, channel),
            set_cmd = partial(controller.set_channel_dacvalue, channel),
            get_parser = BaspiLnhrdac2aController.dacval_to_vval,
            set_parser = BaspiLnhrdac2aController.vval_to_dacval,
            vals = validate.Numbers(min_value = -10.0, max_value = 10.0),
            initial_value = 0.0
        )

        self.high_bandwidth = self.add_parameter(
            name = "high_bandwidth",
            get_cmd = partial(controller.get_channel_bandwidth, channel),
            set_cmd = partial(controller.set_channel_bandwidth, channel),
            val_mapping = create_on_off_val_mapping(on_val = "HBW", off_val = "LBW"),
            initial_value = False
        )

        self.enable = self.add_parameter(
            name = "enable",
            get_cmd = partial(controller.get_channel_status, channel),
            set_cmd = partial(controller.set_channel_status, channel),
            val_mapping = create_on_off_val_mapping(on_val = "ON", off_val = "OFF"),
            initial_value = False
        )


# class ----------------------------------------------------------------

class BaspiLnhrdac2aAWG(InstrumentModule):
    
    def __init__(self, 
                 parent: VisaInstrument, 
                 name: str, 
                 awg: str, 
                 controller: BaspiLnhrdac2aController):
        """
        Class which defines an AWG (Arbitrary Waveform Generator) of the LNHR DAC II with all its QCoDeS-parameters.

        AWG-Parameters:
        channel (1 ... 12 or 13 ... 24, selecting AWG output)
        cycles (0 ... 4 000 000 000, amount of times the waveform is repeated)
        sampling_rate (0.000 01 s ... 4 000 s)
        length (0 ... 65 000, amount of data points)
        time_axis (gets automatically created, depending on AWG settings)
        waveform (-10.000000 V ... +10.000000 V)
        trigger (disable: no external trigger, start only: external trigger starts AWG waveform, 
                start stop: AWG is started by a positive signal edge and stopped by a negative signal edge, 
                single step: positive signal edge triggers every point of the waveform)
        enable (ON/True: start AWG, OFF/False: stop AWG)

        Parameters:
        parent: instrument this channel is a part of
        name: name of the channel
        awg: AWG designator
        controller: the controller the instrument uses for its communication
        """

        super().__init__(parent, name)
        self.__controller = controller

        self.locked = False

        self.channel = self.add_parameter(
            name = "channel",
            get_cmd = partial(controller.get_awg_channel, awg),
            set_cmd = partial(controller.set_awg_channel, awg),
            vals = validate.MultiTypeAnd(
                validate.Ints(min_value = 1, max_value = 24), 
                BaspiLnhrdac2aLockingValidator(self)
            )
        )

        self.cycles = self.add_parameter(
            name = "cycles",
            get_cmd = partial(controller.get_awg_cycles, awg),
            set_cmd = partial(controller.set_awg_cycles, awg),
            vals = validate.MultiTypeAnd(
                validate.Ints(min_value = 0, max_value = 4000000000),
                BaspiLnhrdac2aLockingValidator(self)
            ),
            initial_value = 0
        )

        self.sampling_rate = self.add_parameter(
            name = "sampling_rate",
            unit = "s",
            get_cmd = partial(controller.get_awg_clock_period, awg), #this changed from board to awg, since each awg is now decoupled from the other awgs
            set_cmd = partial(controller.set_awg_clock_period, awg), #this changed from board to awg, since each awg is now decoupled from the other awgs
            get_parser = self.__get_parser_awg_sampling_rate,
            set_parser = self.__set_parser_awg_sampling_rate,
            vals = validate.MultiTypeAnd(
                validate.Numbers(min_value = 0.00001, max_value = 4000.0),
                BaspiLnhrdac2aLockingValidator(self)
            )
        )

        self.length = self.add_parameter(
            # Qcodes only value, not saved on device
            # must be set whenever self.waveform is set
            name = "length",
            get_cmd = None,
            set_cmd = None,
            initial_value = 0,
            vals = validate.MultiTypeAnd(
                validate.Ints(min_value = 0, max_value = 65000),
                BaspiLnhrdac2aLockingValidator(self)
            )
        )

        self.time_axis = self.add_parameter(
            name = "time_axis",
            label = "time",
            unit = "s",
            get_cmd = partial(self.__get_awg_time_axis, awg),
            get_parser = partial(array, dtype = float),
            vals = validate.Arrays(shape = (self.length,))
        )

        self.waveform = self.add_parameter(
            name = "waveform",
            label = f"waveform AWG {awg.upper()}",
            unit = "V",
            parameter_class = ParameterWithSetpoints,
            get_cmd = partial(self.__get_awg_waveform, awg),
            set_cmd = partial(self.__set_awg_waveform, awg),
            get_parser = partial(array, dtype = float),
            set_parser = list,
            setpoints = (self.time_axis,),
            vals = validate.Arrays(shape = (self.length,), min_value = -10.0, max_value = 10.0)
        )

        self.trigger = self.add_parameter(
            name = "trigger",
            get_cmd = partial(controller.get_awg_trigger_mode, awg),
            set_cmd = partial(controller.set_awg_trigger_mode, awg),
            val_mapping = {"disable": 0, "start only": 1, "start stop": 2, "single step": 3},
            vals = BaspiLnhrdac2aLockingValidator(self),
            initial_value = "disable"
        )

        self.enable = self.add_parameter(
            name = "enable",
            get_cmd = partial(controller.get_awg_run_state, awg),
            set_cmd = partial(controller.set_awg_start_stop, awg),
            get_parser = BaspiLnhrdac2aAWG.__get_parser_awg_enable,
            val_mapping = create_on_off_val_mapping(on_val = "START", off_val = "STOP"),
            vals = BaspiLnhrdac2aLockingValidator(self),
            initial_value = False
        )

        self.cycles_done = self.add_parameter(
            name = "cycles_done",
            get_cmd = partial(controller.get_awg_cycles_done, awg),
            get_parser = int,
            label = "Completed AWG cycles"
        )

    #-------------------------------------------------

    @staticmethod
    def __get_parser_awg_sampling_rate(val: int) -> float:
        """
        Parsing method to convert the AWG sampling rate from us (micro seconds) to s (seconds).
        """

        return round(val / 1000000, 6)
    
    #-------------------------------------------------

    @staticmethod
    def __set_parser_awg_sampling_rate(val: float) -> int:
        """
        Parsing method to convert the AWG sampling rate from s (seconds) to us (micro seconds).
        """
        
        return int(val * 1000000)
    
    #-------------------------------------------------

    def __get_awg_time_axis(self, awg: str) -> list[float]:
        """
        Automatically creates the time axis for the saved waveform.

        Parameters:
        awg: selected AWG 

        returns:
        list: list of time values for each voltage saved in the AWG waveform in s
        """

        memory_size = self.__controller.get_wav_memory_size(awg)
        clock_period = self.__controller.get_awg_clock_period(awg)

        increment = clock_period / 1000000
        time_axis = []
        for index in range(0, memory_size):
            time_axis.append(round(index*increment,6))

        return time_axis

    #-------------------------------------------------
        
    def __get_awg_waveform(self, awg: str) -> list[float]:
        """
        Read the AWG waveform from device memory.

        Parameters:
        awg: selected AWG

        Returns:
        list: AWG waveform values in V (Volt)
        """

        memory = []
        block_size = 1000 # number of points read by get_wav_memory_block()
        memory_size = self.__controller.get_wav_memory_size(awg)
        adress_range_limit = memory_size // block_size
        if memory_size % block_size != 0:
            adress_range_limit += 1

        # read memory blocks (1000 points) instead of single adresses for faster reading
        for address in range(0, adress_range_limit):
            data = self.__controller.get_wav_memory_block(awg, address * block_size)
            last_value = data.pop()
            while last_value == "NaN":
                last_value = data.pop()
            data.append(last_value)
            memory.extend(data)

        if len(memory) != memory_size:
            raise MemoryError("Error occured while reading the devices memory.")   
        
        return memory

    #-------------------------------------------------

    def __set_awg_waveform(self, awg: str, waveform: list[float]) -> None:
        """
        Write an AWG waveform into device memory. Memory is cleared before writing.

        Parameters:
        awg: selected AWG
        waveform: list of voltages (+/- 10.000000 V)
        """

        # check for lock
        validator = BaspiLnhrdac2aLockingValidator(self)
        validator.validate(waveform)

        # snapshot parameters that write_wav_to_awg silently resets
        clock_period  = self.__controller.get_awg_clock_period(awg)
        cycles_count  = self.__controller.get_awg_cycles(awg)

        self.__controller.clear_wav_memory(awg)

        for address in range(0, len(waveform)):
            self.__controller.set_wav_memory_value(awg, address, float(waveform[address]))

        sleep(0.2) # sleep bc bad firmware
        memory_size = self.__controller.get_wav_memory_size(awg)

        if len(waveform) != memory_size:
            raise MemoryError("Error occured while writing to the devices memory.")
        
        self.__controller.write_wav_to_awg(awg)
        while self.__controller.get_wav_memory_busy(awg):
            pass

        # reset clock period bc gets changed by a ghost while write_to_wav
        if self.__controller.get_awg_clock_period(awg) != clock_period:
            self.__controller.set_awg_clock_period(awg, clock_period)
        if self.__controller.get_awg_cycles(awg) != cycles_count:
            self.__controller.set_awg_cycles(awg, cycles_count)
    
    #-------------------------------------------------

    @staticmethod
    def __get_parser_awg_enable(val: bool) -> str:
        """
        Parsing method for the parameter "enable". Ensures correct function of val_mapping = create_on_off_val_mapping().
        Output of enable.get() has to be a valid input of enable.set().
        """

        if val: return "START"
        else: return "STOP"
    

# class ----------------------------------------------------------------

@dataclass
class BaspiLnhrdac2aSWGConfig():
    """
    Dataclass to pass a configuration of the LNHR DAC II SWG module.

    Properties: 
    shape: "sine", "cosine, "triangle", "sawtooth", "ramp", "rectangle", "pulse", "fixed noise", "random noise" or "DC"
    frequency: signal frequency in Hz (0.001 Hz - 10000 Hz)
    amplitude: signal amplitude in V (+/- 10.000 V)
    offset: signal DC-offset (+/- 10.000 V)
    phase: signal phaseshift in ° (deg) (+/- 360.000°)
    dutycycle: signal dutycycle in % (0.0 - 100.0), only applicable with shape "pulse"
    """

    shape: str
    frequency: float
    amplitude: float
    offset: float
    phase: float
    dutycycle: float

    #-------------------------------------------------

    def __post_init__(self):
        """default values for unspecified values"""
        if isinstance(self.shape, property):
            self.shape = "sine"
        if isinstance(self.frequency, property):
            self.frequency = 100.0
        if isinstance(self.amplitude, property):
            self.amplitude = 1.0
        if isinstance(self.offset, property):
            self.offset = 0.0
        if isinstance(self.phase, property):
            self.phase = 0.0
        if isinstance(self.dutycycle, property):
            self.dutycycle = 0.0

    #-------------------------------------------------

    def __check_min_max(self, val: int | float, min: int | float, max: int | float, prop: str) -> None:
        """check validity of properties"""
        if isinstance(val, property):
            # do nothing if value is not specified
            return
        
        if not isinstance(val, (int, float)):
            raise ValueError(f"Configuration value {prop} is of not the correct type.")
        if val < min: 
            raise ValueError(f"Configuration value {prop} is too small. Increase {prop} to {min}.")
        if val > max: 
            raise ValueError(f"Configuration value {prop} is too big. Decrease {prop} to {max}.")
        
    #-------------------------------------------------

    @property
    def frequency(self) -> float:
        return self._frequency
    @frequency.setter
    def frequency(self, val: float) -> None:
        self.__check_min_max(val, min = 0.001, max = 10_000.0, prop = "frequency")
        self._frequency = val

    @property
    def amplitude(self) -> float:
        return self._amplitude
    @amplitude.setter
    def amplitude(self, val: float) -> None:
        self.__check_min_max(val, min = -50.0, max = 50.0, prop = "amplitude")
        self._amplitude = val

    @property
    def offset(self) -> float:
        return self._offset
    @offset.setter
    def offset(self, val: float) -> None:
        self.__check_min_max(val, min = -10.0, max = 10.0, prop = "offset")
        self._offset = val
    
    @property
    def phase(self) -> float:
        return self._phase
    @phase.setter
    def phase(self, val: float) -> None:
        self.__check_min_max(val, min = -360.0, max = 360.0, prop = "phase")
        self._phase = val

    @property
    def dutycycle(self) -> float:
        return self._dutycycle
    @dutycycle.setter
    def dutycycle(self, val: float) -> None:
        self.__check_min_max(val, min = 0.0, max = 100.0, prop = "dutycycle")
        self._dutycycle = val
            

# class ----------------------------------------------------------------

class BaspiLnhrdac2aSWG(InstrumentModule):

    def __init__(self, parent: VisaInstrument, name: str, controller: BaspiLnhrdac2aController):
        """
        Class defining the Standard Waveform Generator (SWG) module of the LNHR DAC II with all its Qcodes Parameters.
        The SWG can be used to create a waveform, which is then outputted by an AWG.

        SWG-Parameters:
        configuration (object of type BaspiLnhrdac2aSWGConfig, to configure the SWG)
        apply (A, B, C, D, E, or F applies the configured waveform and saves it to the AWG A, B, C, D, E or F)

        Parameters:
        parent: instrument this channel is a part of
        name: name of the module
        controller: the controller the instrument uses for its communication
        """

        super().__init__(parent, name)
        self.__controller = controller

        self.configuration = self.add_parameter(
            name = "configuration",
            get_cmd = None,
            set_cmd = self.__set_swg_configuration
        )
    
    #-------------------------------------------------

    def __set_swg_configuration(self, config: BaspiLnhrdac2aSWGConfig) -> None:
        """
        Create a waveform using the standard waveform generator. The resulting waveform is automatically written into the waveform memory.

        config-Attributes:
        shape: "sine", "cosine, "triangle", "sawtooth", "ramp", "rectangle", "pulse", "fixed noise", "random noise" or "DC"
        frequency: signal frequency in Hz (0.001 Hz - 10000 Hz)
        amplitude: signal amplitude in V (+/- 10.000 V)
        offset: signal DC-offset (+/- 10.000 V)
        phase: signal phaseshift in ° (deg) (+/- 360.000°)
        dutycycle: signal dutycycle in % (0.0 - 100.0), only applicable with shape "pulse"

        Parameters:
        awg: selected AWG
        config: object containing SWG configuration
        """
        
        self.__controller.set_swg_new(True)

        # always use "adapt clock" here, clock gets checked again in swg.apply
        self.__controller.set_swg_adapt_clock(True)

        awg_shapes = {"sine": 0,
                      "cosine": 0,
                      "triangle": 1,
                      "sawtooth": 2,
                      "ramp": 3,
                      "rectangle": 4,
                      "pulse": 4,
                      "fixed noise": 5,
                      "random noise": 6,
                      "DC": 7}

        if config.shape not in awg_shapes:
            raise ValueError(f"Value '{config.shape}' is invalid. Valid values are: {list(awg_shapes.keys())}.")
        
        # specify waveform
        self.__controller.set_swg_shape(awg_shapes[config.shape])
        self.__controller.set_swg_desired_frequency(config.frequency)
        self.__controller.set_swg_amplitude(config.amplitude)
        self.__controller.set_swg_offset(config.offset)

        if config.shape == "cosine":
            self.__controller.set_swg_phase(config.phase + 90.0)
        else:
            self.__controller.set_swg_phase(config.phase)
        if config.shape == "rectangle":
            self.__controller.set_swg_dutycycle(50.0)
        elif config.shape == "pulse":
            self.__controller.set_swg_dutycycle(config.dutycycle)

    #-------------------------------------------------

    def apply(self, awg: str) -> None:
        """
        Apply the SWG configuration to an AWG waveform.

        Parameters:
        awg: selected AWG
        """

        awg = awg.lower()

        # set awgX.length parameter, also checks if AWG is locked
        wav_memory_size = self.__controller.get_wav_memory_size(awg)
        getattr(self.parent, f"awg{awg}").length.set(wav_memory_size)

        self.__controller.set_swg_wav_memory(awg)
        # each awg has its own independent clock with the DAC IIa
        self.__controller.set_swg_adapt_clock(True)

        desired_frequency = self.__controller.get_swg_desired_frequency()
        nearest_frequency = self.__controller.get_swg_nearest_frequency()
        if nearest_frequency != desired_frequency:
            print(f"Frequency of {desired_frequency} Hz cannot be reached with the current settings. "
                + f"A frequency of {nearest_frequency} Hz is used instead. "
                + f"Changing AWG or clearing unused AWG waveforms might resolve this issue.")

        # apply SWG configuration to AWG waveform
        self.__controller.apply_swg_operation()
        self.__controller.write_wav_to_awg(awg)
        while self.__controller.get_wav_memory_busy(awg):
            pass

        awg_memory_size = self.__controller.get_awg_memory_size(awg)
        if awg_memory_size != wav_memory_size:
            getattr(self.parent, f"awg{awg}").length.set(awg_memory_size)   


# class ----------------------------------------------------------------

@dataclass
class BaspiLnhrdac2aFast2dConfig():
    """
    Dataclass to pass a configuration of the LNHR DAC II Fast Scan 2D module.

    Properties:
    scan_unit: AWG which will be used for the 2D-scan (a, b, c, d with the corresponding STEP-generator)
    x_channel: channel of the x-axis (1 - 12)
    x_start_voltage: starting voltage of the x-axis in V (+/- 10.000000 V)
    x_stop_voltage: ending voltage of the x-axis in V (+/- 10.000000 V)
    x_steps: number of steps the x-axis voltage is incremented
    y_channel: channel of the y-axis (1 - 12)
    y_start_voltage: starting voltage of the x-axis in V (+/- 10.000000 V)
    y_stop_voltage: ending voltage of the x-axis in V (+/- 10.000000 V)
    y_steps: number of steps the y-axis voltage is incremented
    acquisition_delay: time for which each voltage step is outputted in s
    adaptive_shift: voltage shift in V which is applied to the x-axis, after every y-axis sweep (+/- 10.000000 V)
    """
    scan_unit: str
    x_channel: int
    x_start_voltage: float
    x_stop_voltage: float
    x_steps: int
    y_channel: int
    y_start_voltage: float
    y_stop_voltage: float
    y_steps: int
    acquisition_delay: float
    adaptive_shift: float

    #-------------------------------------------------

    def __post_init__(self):
        """default values for unspecified values"""
        if isinstance(self.scan_unit, property):
            self.scan_unit = "a"
        if isinstance(self.x_channel, property):
            self.x_channel = 1
        if isinstance(self.x_start_voltage, property):
            self.x_start_voltage = 0.0
        if isinstance(self.x_stop_voltage, property):
            self.x_stop_voltage = 1.0
        if isinstance(self.x_steps, property):
            self.x_steps = 10
        if isinstance(self.y_channel, property):
            self.y_channel = 2
        if isinstance(self.y_start_voltage, property):
            self.y_start_voltage = 0.0
        if isinstance(self.y_stop_voltage, property):
            self.y_stop_voltage = 1.0
        if isinstance(self.y_steps, property):
            self.y_steps = 10
        if isinstance(self.acquisition_delay, property):
            self.acquisition_delay = 0.00001
        if isinstance(self.adaptive_shift, property):
            self.adaptive_shift = 0.0
        
        self._validate_channels()
    
    #-------------------------------------------------
    def _validate_channels(self) -> None:
        """validate that channels match the scan_unit's board"""
        if isinstance(self.scan_unit, property):
            return
        
        scan_unit_lower = self.scan_unit.lower()

        # determine valid channel range
        if scan_unit_lower in ("a", "b"):
            valid_range = (1, 12)
            board_name = "lower"
        elif scan_unit_lower in ("c", "d"):
            valid_range = (13, 24)
            board_name = "upper"
        else:
            raise ValueError(f"scan_unit must be 'a', 'b', 'c', 'd', got '{self.scan_unit}'.")
        
        # validate y_channel (makeing sure it matches the AWG-board)
        if not isinstance(self.y_channel, property):
            if not (valid_range[0]<= self.y_channel <= valid_range[1]):
                raise ValueError(f"y_channel must be {valid_range[0]}-{valid_range[1]} for scan unit '{scan_unit_lower}' ({board_name} board), got {self.y_channel}.")
        
        # ensure x and y use different channels
        if (not isinstance(self.x_channel, property) and not isinstance(self.y_channel, property)):
            if self.x_channel == self.y_channel:
                raise ValueError(f"x_channel and y_channel must be differen (both are set to {self.x_channel})!")

    #-------------------------------------------------
    def __check_min_max(self, val: int | float, min: int | float, max: int | float, prop: str) -> None:
        """check validity of properties"""
        if isinstance(val, property):
            # default values are not checked!
            return
        
        if not isinstance(val, (int, float)):
            raise ValueError(f"Configuration value {prop} is of not the correct type.")
        if val < min: 
            raise ValueError(f"Configuration value {prop} is too small. Increase {prop} to {min}.")
        if val > max: 
            raise ValueError(f"Configuration value {prop} is too big. Decrease {prop} to {max}.")
        
    #-------------------------------------------------
    def __check_gen(self, val: str, allowed: tuple[str, ...], prop: str) -> None:
        """Check validity of string properties"""
        if isinstance(val, property):
            return  # Default values are not checked
        
        if not isinstance(val, str):
            raise ValueError(f"Configuration value {prop} is not the correct type.")
        if val.lower() not in allowed:
            raise ValueError(
                f"Configuration value {prop} is invalid. Allowed values: {allowed}."
            )
    
    #-------------------------------------------------

    @property
    def scan_unit(self) -> str:
        return self._scan_unit
    @scan_unit.setter
    def scan_unit(self, val: str):
        self.__check_gen(val, allowed= ("a", "b", "c", "d"), prop= "scan_unit")
        self._scan_unit = val
        
    @property
    def x_channel(self) -> int:
        return self._x_channel
    @x_channel.setter
    def x_channel(self, val: int) -> None:
        self.__check_min_max(val, min = 1, max = 24, prop = "x_channel")
        self._x_channel = val

    @property
    def x_start_voltage(self) -> float:
        return self._x_start_voltage
    @x_start_voltage.setter
    def x_start_voltage(self, val: float) -> None:
        self.__check_min_max(val, min = -10.0, max = 10.0, prop = "x_start_voltage")
        self._x_start_voltage = val

    @property
    def x_stop_voltage(self) -> float:
        return self._x_stop_voltage
    @x_stop_voltage.setter
    def x_stop_voltage(self, val: float) -> None:
        self.__check_min_max(val, min = -10.0, max = 10.0, prop = "x_stop_voltage")
        self._x_stop_voltage = val

    @property
    def x_steps(self) -> int:
        return self._x_steps
    @x_steps.setter
    def x_steps(self, val: int) -> None:
        self.__check_min_max(val, min = 10, max = 16_777_216, prop = "x_steps")
        self._x_steps = val

    @property
    def y_channel(self) -> int:
        return self._y_channel
    @y_channel.setter
    def y_channel(self, val: int) -> None:
        self.__check_min_max(val, min = 1, max = 24, prop = "y_channel")
        self._y_channel = val

    @property
    def y_start_voltage(self) -> float:
        return self._y_start_voltage
    @y_start_voltage.setter
    def y_start_voltage(self, val: float) -> None:
        self.__check_min_max(val, min = -10.0, max = 10.0, prop = "y_start_voltage")
        self._y_start_voltage = val

    @property
    def y_stop_voltage(self) -> float:
        return self._y_stop_voltage
    @y_stop_voltage.setter
    def y_stop_voltage(self, val: float) -> None:
        self.__check_min_max(val, min = -10.0, max = 10.0, prop = "y_stop_voltage")
        self._y_stop_voltage = val

    @property
    def y_steps(self) -> int:
        return self._y_steps
    @y_steps.setter
    def y_steps(self, val: int) -> None:
        self.__check_min_max(val, min = 1, max = 16_777_216, prop = "y_steps")
        self._y_steps = val

    @property
    def acquisition_delay(self) -> float:
        return self._acquisition_delay
    @acquisition_delay.setter
    def acquisition_delay(self, val: float) -> None:
        self.__check_min_max(val, min = 0.00001, max = 4000.0, prop = "acquisition_delay")
        self._acquisition_delay = val

    @property
    def adaptive_shift(self) -> float:
        return self._adaptive_shift
    @adaptive_shift.setter
    def adaptive_shift(self, val: float) -> None:
        self.__check_min_max(val, min = -10.0, max = 10.0, prop = "adaptive_shift")
        self._adaptive_shift = val

      
# class ----------------------------------------------------------------

class BaspiLnhrdac2aFast2d(InstrumentModule):

    def __init__(self, 
                 parent: VisaInstrument, 
                 name: str, 
                 controller: BaspiLnhrdac2aController):
        """
        Class which defines an adaptive fast 2D-scan of the LNHR DAC IIa with all its QCoDeS-parameters.

        The 2D scan uses one STEP/AWG pair (A, B, C, or D):
        STEP-generator: x-axis (slow axis)
        AWG: y-axis (fast axis)

        2D-scan-Parameters:
        configuration (object of type BaspiLnhrdac2aFast2dConfig, to configure the fast 2D-scan)
        trigger (disable: no trigger/ scan as fast as possible, line in: external trigger starts every x-axis sweep, 
                line out: trigger is set with every x-axis sweep, point out: trigger is set with every x-axis step)
        x_axis (voltages in V of x-axis sweep, only gettable)
        y_axis (voltages in V of y-axis sweep, only gettable)
        enable (ON/True: start fast 2D-scan, OFF/False: stop fast 2D-scan)

        Parameters:
        parent: instrument this channel is a part of
        name: name of the channel
        controller: the controller the instrument uses for its communication
        """

        super().__init__(parent, name)
        self.__controller = controller

        self.__scan_configs = {}
        self.__trigger_modes = {}
        self.__current_scan = None

        self.configuration = self.add_parameter(
            name = "configuration",
            get_cmd = None,
            set_cmd = self.__set_2d_configuration
        )

        self.trigger = self.add_parameter(
            name = "trigger",
            get_cmd = self.__get_2d_trigger,
            set_cmd = self.__set_2d_trigger,
        )

        self.x_axis = self.add_parameter(
            name = "x_axis",
            unit = "V",
            get_cmd = self.__get_2d_x_axis,
            set_cmd = None
        )

        self.y_axis = self.add_parameter(
            name = "y_axis",
            unit = "V",
            get_cmd = self.__get_2d_y_axis,
            set_cmd = None
        )

        self.enable = self.add_parameter(
            name = "enable",
            get_cmd = None,
            set_cmd = self.__set_2d_enable,
            val_mapping = create_on_off_val_mapping(on_val = True, off_val = False),
            initial_value = False
        )

    #-------------------------------------------------
    def select_scan(self, scan_unit: str) -> None:
        """
        Select which scan to control with parameters.
        
        After calling this, all parameter operations (.get(), .set()) 
        will operate on the selected scan.
        
        Parameters:
        scan_unit: which scan to select ("a", "b", "c", or "d")
        """
        scan_unit = scan_unit.lower()
        
        if scan_unit not in ("a", "b", "c", "d"):
            raise ValueError(f"Invalid scan_unit '{scan_unit}'. Must be 'a', 'b', 'c', or 'd'.")
        
        self.__current_scan = scan_unit
        print(f"Selected scan: STEP/AWG-{scan_unit.upper()}")
        
        if scan_unit not in self.__scan_configs:
            print(f"Warning: Scan {scan_unit.upper()} not yet configured")

    #-------------------------------------------------

    def __set_2d_configuration(self, config: BaspiLnhrdac2aFast2dConfig) -> None: 
        """
        Create an adaptive fast 2D-scan.

        config-Attributes:
        scan_unit: AWG which will be used for the 2D-scan (a, b, c, d with the corresponding STEP-generator)
        x_channel: channel of the x-axis (1 - 12)
        x_start_voltage: starting voltage of the x-axis in V (+/- 10.000000 V)
        x_stop_voltage: ending voltage of the x-axis in V (+/- 10.000000 V)
        x_steps: number of steps the x-axis voltage is incremented
        y_channel: channel of the y-axis (1 - 12)
        y_start_voltage: starting voltage of the x-axis in V (+/- 10.000000 V)
        y_stop_voltage: ending voltage of the x-axis in V (+/- 10.000000 V)
        y_steps: number of steps the y-axis voltage is incremented
        acquisition_delay: time for which each voltage step is outputted in s
        adaptive_shift: voltage shift in V which is applied to the x-axis, after every y-axis sweep (+/- 10.000000 V)

        Parameters:
        config: object containing 2D scan configuration
        """
        
        scan_unit = config.scan_unit.lower()
        scan_unit_upper = config.scan_unit.upper()

        print(f"Starting to configure fast adaptive 2D scan. AWG/STEP-{scan_unit_upper} will be repurposed. AWG/STEP-{scan_unit_upper} cannot be used while the 2D scan is running.")

        # check if AWG and STEP-generator can be used
        if (not self.__controller.get_awg_run_state(scan_unit) and
            self.__controller.get_ramp_state(scan_unit) == 0):
            #self.__scan_unit = config.scan_unit.lower()
            pass
        else:
            raise SystemError(f"STEP-{scan_unit_upper} or AWG-{scan_unit_upper} are currently running. Both must be idle for 2D scan setup.")
        
        awg = getattr(self.parent, f"awg{scan_unit}")
        awg.locked = False

        # self.__controller.set_awg_channel(self.__awg_xy, config.y_channel)
        awg.channel.set(config.y_channel)
        if not self.__controller.get_awg_channel_availability(scan_unit):
            raise SystemError(f"The chosen y-axis output (channel {config.y_channel}) is not available.")
        
        self.__controller.set_ramp_channel(scan_unit, config.x_channel)
        if not self.__controller.get_ramp_channel_availability(scan_unit):
            raise SystemError(f"The chosen x-axis output (channel {config.x_channel}) is not available.")

        # calculate internal values, check for limits
        x_ramp_time = 0.005 * (config.x_steps + 1)
        y_step_size = (config.y_stop_voltage - config.y_start_voltage) / config.y_steps
        y_period = config.y_steps * config.acquisition_delay
        if y_period < 0.006:
            raise SystemError(f"The configured y-axis sweep is too short ({y_period:.3f} s). Minimal sweep time is 0.006 s. Increase number of steps or acquisition delay.")

        # set up x-axis
        self.__controller.set_ramp_starting_voltage(scan_unit, config.x_start_voltage)
        self.__controller.set_ramp_peak_voltage(scan_unit, config.x_stop_voltage)
        self.__controller.set_ramp_duration(scan_unit, x_ramp_time)
        self.__controller.set_ramp_shape(scan_unit, 0)
        self.__controller.set_ramp_cycles(scan_unit, 1)
        self.__controller.select_ramp_step(scan_unit, 1)

        # set up y-axis
        y_axis_waveform = []
        for step in range(0, config.y_steps + 1):
            y_axis_waveform.append((step * y_step_size) + config.y_start_voltage)
        y_axis_waveform.append(config.y_start_voltage)
        y_axis_waveform = array(y_axis_waveform)

        
        awg.trigger.set("disable")
        awg.cycles.set(1)
        awg.sampling_rate.set(config.acquisition_delay)
        awg.length.set(len(y_axis_waveform))
        awg.waveform.set(y_axis_waveform)

        # set up adaptive shift
        adaptive_scan = 1 if config.adaptive_shift != 0.0 else 0
        self.__controller.set_awg_start_mode(scan_unit, 1)
        self.__controller.set_awg_reload_mode(scan_unit, adaptive_scan)
        self.__controller.set_apply_polynomial(scan_unit, adaptive_scan)

        # lock AWG to prevent User from manipulating/ breaking stuff
        awg.locked = True

        self.__scan_configs[scan_unit] = config

        self.__current_scan = scan_unit
        

        print("Fast adaptive 2D scan sucessfully configured. Ready to start.")

    #-------------------------------------------------

    def __set_2d_trigger(self, mode: str) -> None:
        """
        Set the trigger mode of fast 2D-scan.
         
        Parameters:
        mode: "disable": no trigger/ scan as fast as possible, "line in": external trigger starts every x-axis sweep, 
                "line out": trigger is set with every x-axis sweep, "point out": trigger is set with every x-axis step
       
        """
        if self.__current_scan is None:
            raise RuntimeError("No scan selected. Use 'select_scan('a')' first or configure a scan")
        
        
        scan_unit = self.__current_scan
        scan_unit_upper = scan_unit.upper()
        
        print("Starting to configure fast 2D scan trigger.")
        
        fast2d_triggers = (
            "disable",
            "line in",
            "line out",
            "point out"
        )

        if mode not in fast2d_triggers:
            raise ValueError(f"Value '{mode}' is invalid. Valid values are: {fast2d_triggers}.")

        if scan_unit not in self.__scan_configs:
            raise SystemError(f"No configuration for scan_unit '{scan_unit}'. Use configuration.set() first")
        
        awg = getattr(self.parent, f"awg{scan_unit}")
               

        print(f"Configurating 2D scan trigger mode for {scan_unit}: '{mode}'...")
        if mode == "disable":
            awg.locked = False
            awg.trigger.set("disable")
            self.__controller.set_awg_start_mode(scan_unit, 1)
            awg.locked = True
            print(f"Fast 2D scan trigger disabled")

        elif mode == "line in":
            awg.locked = False
            awg.trigger.set("start only")
            self.__controller.set_awg_start_mode(scan_unit, 0)
            awg.locked = True
            print(f"Fast 2D scan trigger set to '{mode}'.")

        elif mode == "line out":
            awg.locked = False
            awg.trigger.set("disable")
            self.__controller.set_awg_start_mode(scan_unit, 1)
            awg.locked = True
            print(f"Fast 2D scan trigger set to '{mode}'.")

        elif mode == "point out":
            awg.locked = False
            awg.trigger.set("disable")
            self.__controller.set_awg_start_mode(scan_unit, 1)
            awg.locked = True
            print(f"Fast 2D scan trigger set to '{mode}'.")

        self.__trigger_modes[scan_unit] = mode
    #-------------------------------------------------
    def __get_2d_trigger(self) -> str:
        """
        Get the trigger mode of currently selected scan.

        Returns:
        str: trigger mode or "not set" if no scan selected
        """
        if self.__current_scan is None:
            return "disable"
    
        return self.__trigger_modes.get(self.__current_scan, "disable")

    #-------------------------------------------------

    def __get_2d_x_axis(self) -> ndarray:
        """
        Get the x-axis voltage steps which are outputted in a x-axis sweep.

        Returns:
        ndarray: numpy array with voltage steps in V (+/- 10.000000 V)
        """

        if self.__current_scan is None or self.__current_scan not in self.__scan_configs:
            return array([], dtype = float)
        
        scan_unit = self.__current_scan
        
        step_size = self.__controller.get_ramp_step_size(scan_unit)
        number_steps = self.__controller.get_ramp_cycle_steps(scan_unit)
        start_voltage = self.__controller.get_ramp_starting_voltage(scan_unit)

        waveform = []
        for step in range(0, number_steps):
            waveform.append(round(start_voltage + (step * step_size), 6))

        return array(waveform, dtype= float)
        
    #-------------------------------------------------

    def __get_2d_y_axis(self) -> ndarray:
        """
        Get the y-axis voltage steps which are outputted in a y-axis sweep.

        Returns:
        ndarray: numpy array with voltage steps in V (+/- 10.000000 V)
        """
        if self.__current_scan is None or self.__current_scan not in self.__scan_configs:
            return array([], dtype = float)
                
        scan_unit = self.__current_scan

        #Guard: require a config (prevents partial reuse)
        awg = getattr(self.parent, f"awg{scan_unit}")

        was_locked = awg.locked
        awg.locked = False
        waveform = awg.waveform.get()
        awg.locked = was_locked

        #delete last element (returns to starting value)
        waveform = list(waveform)
        if len(waveform) > 0:
            waveform.pop()
        
        return array(waveform, dtype=float) 
    #-------------------------------------------------


    def __set_2d_enable(self, enable: bool) -> None:
        """
        Start or stop the fast 2D-scan by software.

        Parameters:
        enable: start or stop 2D-scan
        """
        if self.__current_scan is None:
            if enable:
                raise RuntimeError("No scan selected. Use 'select_scan('a')' first or configure a scan")
            return
        
        scan_unit = self.__current_scan
        scan_unit_upper = scan_unit.upper()

        if scan_unit not in self.__scan_configs:
            if enable:
                raise SystemError(f"Cannot start scan: no configuration was set for scan_unit '{scan_unit}'. Use configuration.set() first")
            return
        
        
        config = self.__scan_configs[scan_unit]
        awg = getattr(self.parent, f"awg{scan_unit}")

        if enable:
            # sanity check again
            if self.__controller.get_awg_run_state(scan_unit):
                raise SystemError(f"Cannot start: AWG {scan_unit_upper} is already running")
            
            awg.locked = False
            awg.enable.set(True)
            awg.locked = True

            print(f"Fast adaptive 2D scan {scan_unit_upper} started with configuration {config}")
        
        else:
            awg.locked = False
            awg.enable.set(False)
            awg.locked = False
            #resets config, so partially reuse is not possible
            print(f"2D scan on AWG-{scan_unit_upper} stopped")
            del self.__scan_configs[scan_unit]
            if scan_unit in self.__trigger_modes:
                del self.__trigger_modes[scan_unit]
                
    #--------------------------------------------------------------------   
    def list_scans(self)-> None:
        """Display status of all configured scans."""
        if not self.__scan_configs:
            print("\nNo 2D scans configured.\n")
            return
        
        current_indicator = f" (current: {self.__current_scan.upper()})" if self.__current_scan else ""
        print(f"\n=== Configured 2D Scans ({len(self.__scan_configs)}){current_indicator} ===")
        
        for scan_unit, config in self.__scan_configs.items():
            scan_unit_upper = scan_unit.upper()
            running = self.__controller.get_awg_run_state(scan_unit)
            status = "RUNNING" if running else "READY"
            trigger = self.__trigger_modes.get(scan_unit, "not set")
            current = " <- CURRENT" if scan_unit == self.__current_scan else ""
            
            print(f"\nSTEP/AWG-{scan_unit_upper}: {status}{current}")
            print(f"  X: Ch{config.x_channel} ({config.x_start_voltage}V -> {config.x_stop_voltage}V, {config.x_steps} steps)")
            print(f"  Y: Ch{config.y_channel} ({config.y_start_voltage}V -> {config.y_stop_voltage}V, {config.y_steps} steps)")
            print(f"  Points: {config.x_steps * config.y_steps}, Trigger: {trigger}")

    def start_all(self) -> None:
        """Start all configured scans."""
        if not self.__scan_configs:
            print("No scans configured to start.")
            return
        
        for scan_unit in self.__scan_configs.keys():
            self.select_scan(scan_unit)
            self.enable.set(True)

    def stop_all(self) -> None:
        """Stop all running scans."""
        if not self.__scan_configs:
            print("No scans configured.")
            return
        
        for scan_unit in list(self.__scan_configs.keys()):
            self.select_scan(scan_unit)
            self.enable.set(False)               

# class ----------------------------------------------------------------

class BaspiLnhrdac2aBoard(InstrumentModule):

    def __init__(self, parent: VisaInstrument, name: str, board: str, controller: BaspiLnhrdac2aController):
        """
        Class which defines a Board of the LNHR DAC IIa with all its QCoDeS-parameters.
        
        Each board contains:
        - DAC channels (lower: 1-12, upper: 13-24)
        - AWGs (lower: A/B/E, upper: C/D/F)
        
        Board Parameters:
            awg_only_mode: AWG mode ("normal": all channels available, 
                          "awg_only": only AWG channels active with lower jitter)
            sync_mode: DAC update mode ("instant": immediate update, 
                      "synchronous": wait for sync trigger)
        
        Board Methods:
            start_awgs(): Start all AWGs on this board synchronously
            stop_awgs(): Stop all AWGs on this board
            sync(): Trigger synchronous update of all DAC channels on this board

        Parameters:
        parent: instrument this module is part of
        name: name of the module
        board: board designator ("lower" or "upper")
        controller: the controller the instrument uses for its communication
        """

        super().__init__(parent, name)
        self.__controller = controller

        # map board name to hardware identifiers
        if board.lower() == "lower":
            self.__awg_board = "ABE"
            self.__dac_board = "L"
            self.__awgs = ("a", "b", "e")
        elif board.lower() == "upper":
            self.__awg_board = "CDF"
            self.__dac_board = "H"
            self.__awgs = ("c", "d", "f")   
        else:
            raise ValueError(f"Invalid board '{board}'. Must be 'lower' or 'upper'.")   
        
        # AWG board mode parameter
        self.awg_only_mode = self.add_parameter(
            name = "awg_only_mode",
            get_cmd = partial(controller.get_awg_board_mode, self.__awg_board),
            set_cmd = self.__set_awg_only_mode,
            get_parser = int,
            val_mapping = {"normal": 0, "awg_only": 1}
        )

        # DAC synchronization mode parameter
        self.sync_mode = self.add_parameter(
            name = "sync_mode",
            get_cmd = partial(controller.get_board_update_mode, self.__dac_board),
            set_cmd = partial(controller.set_board_update_mode, self.__dac_board),
            val_mapping = {"instant": 0, "synchronous": 1}
        )

    #--------------------------------------------------------------------     

    def __is_any_awg_running(self) -> bool:
        """
        Check if any AWG on this board is currently running.

        Returns:
        bool: True if any AWG is running, False if all are idle
        """

        for awg in self.__awgs:
            if self.__controller.get_awg_run_state(awg):
                return True
        return False

    #-------------------------------------------------------------------- 

    def __set_awg_only_mode(self, mode: int) -> str:
        """
        Set the AWG-only mode of this board, with safety check.

        Parameters:
        mode: normal mode (0) or AWG-only mode (1)

        Returns:
        str: DAC-Error Code

        Raises:
        RuntimeError: if any AWG on this board is currently running
        """

        if self.__is_any_awg_running():
            raise RuntimeError(
                f"Cannot change AWG-only mode while AWGs are running on this board. "
                f"Stop all AWGs first."
            )
        
        return self.__controller.set_awg_board_mode(self.__awg_board, mode)

    #--------------------------------------------------------------------

    def start_awgs(self) -> None:
        """
        Start all AWGs on this board synchronously.
        """

        self.__controller.set_awg_start_stop(self.__awg_board, "START")

    #--------------------------------------------------------------------

    def stop_awgs(self) -> None:
        """
        Stop all AWGs on this board.
        """

        self.__controller.set_awg_start_stop(self.__awg_board, "STOP")

    #--------------------------------------------------------------------

    def sync(self) -> None:
        """
        Trigger synchronous update of all DAC channels on this board.
        
        Board must be in synchronous mode first (sync_mode.set("synchronous")).
        All registered voltages will be applied simultaneously.
        """

        self.__controller.update_board_channels(self.__dac_board)

# class ----------------------------------------------------------------

class BaspiLnhrdac2a(VisaInstrument):
    
    def __init__(self, name: str, address: str):
        """
        Main class for integrating the Basel Precision Instruments 
        LNHR DAC II into QCoDeS as an instrument.

        Parameters:
        name: name of the instrument
        address: VISA address of the instrument
        """

        super().__init__(name, address)

        # "library" of all DAC commands
        # not to be used outside of this class definition
        # to only have a single interface to the device
        self.__controller = BaspiLnhrdac2aController(self)

        # visa properties for communication
        self.visa_handle.write_termination = "\r\n"
        self.visa_handle.read_termination = "\r\n"

        # get number of physicallly available channels
        # for correct further initialization
        channel_modes = self.__controller.get_all_mode()
        self.number_channels = len(channel_modes)
        if self.number_channels != 12 and self.number_channels != 24:
            raise SystemError("Physically available number of channels is not 12 or 24. Please check device.")

        # create channels and add to instrument
        # save references for later grouping
        channels = {}
        for channel_number in range(1, self.number_channels + 1):
            name = f"ch{channel_number}"
            channel = BaspiLnhrdac2aChannel(self, name, channel_number, self.__controller)
            channels.update({name: channel})
            self.add_submodule(name, channel)

        # grouping channels to simplify simoultaneous access
        all_channels = ChannelList(self, "all channels", BaspiLnhrdac2aChannel)
        for channel_number in range(1, self.number_channels + 1):
            channel = channels[f"ch{channel_number}"]
            all_channels.append(channel)

        self.add_submodule("all", all_channels)

        
        lower_board = ChannelList(self, "lower board", BaspiLnhrdac2aChannel)
        for channel_number in range(1, 12 + 1):
            channel = channels[f"ch{channel_number}"]
            lower_board.append(channel)

        self.add_submodule("lower_board", lower_board)
        
        if self.number_channels == 24:
            upper_board = ChannelList(self, "upper board", BaspiLnhrdac2aChannel)
            for channel_number in range(13, 24 + 1):
                channel = channels[f"ch{channel_number}"]
                upper_board.append(channel)

            self.add_submodule("upper_board", upper_board)

        # AWGs dependent on 12/24 channel version
        if self.number_channels == 12:
            awgs = ("a", "b", "e")
        elif self.number_channels == 24:
            awgs = ("a", "b", "c", "d", "e", "f")

        for awg_designator in awgs:
            name = f"awg{awg_designator}"
            awg = BaspiLnhrdac2aAWG(self, name, awg_designator, self.__controller)
            self.add_submodule(name, awg)

        # only one SWG module available
        name = "swg"
        swg = BaspiLnhrdac2aSWG(self, name, self.__controller)
        self.add_submodule(name, swg)

        # only one 2D scan module available
        name = "fast2d"
        fast2d = BaspiLnhrdac2aFast2d(self, name, self.__controller)
        self.add_submodule(name, fast2d)

        # Board control (combines AWG and DAC board features)
        # lower board always exists (both 12 and 24 channel versions)
        name = "board_lower"
        board_lower = BaspiLnhrdac2aBoard(self, name, "lower", self.__controller)
        self.add_submodule(name, board_lower)

        # upper board only for 24 channel version
        if self.number_channels == 24:
            name = "board_upper"
            board_upper = BaspiLnhrdac2aBoard(self, name, "upper", self.__controller)
            self.add_submodule(name, board_upper)

        # global AWG 1 MHz reference clock
        self.awg_refclock = self.add_parameter(
            name = "awg_refclock",
            get_cmd = self.__controller.get_awg_refclock_state,
            set_cmd = self.__controller.set_awg_refclock_state,
            get_parser = BaspiLnhrdac2a.__get_parser_awg_refclock,
            val_mapping = create_on_off_val_mapping(on_val = 1, off_val = 0)
        )

        # display some information after instanciation/ initial connection
        print("")
        self.connect_message()
        print("All channels have been turned off (1 MOhm Pull-Down to AGND) upon initialization "
              + "and are pre-set to 0.0 V if turned on without setting a voltage beforehand.")
        print("")

    #-------------------------------------------------

    def get_idn(self) -> dict:
        """
        Get the identification information of the device.

        Returns:
        dict: contains all QCodes required IDN fields
        """
        vendor = "Basel Precision Instruments AG (BASPI)"
        model = f"LNHR DAC IIa (SP1085) - {self.number_channels} channel version"

        hardware_info = self.__controller.get_serial()
        serial = hardware_info[37:52]
        software_info = self.__controller.get_firmware()
        firmware = software_info[18:33]

        idn = {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware
        }

        return idn

    #-------------------------------------------------    
    
    @staticmethod
    def __get_parser_awg_refclock(val: str) -> int:
        """
        Parsing method for the parameter "awg_reclock".
        Converts "on"/"off" string to integer for val_mapping.
        """
        if val.lower() == "on":
            return 1
        else:
            return 0

    # ------------------------------------------------------------
    def start_all_awgs(self) -> None:
        """
        Docstring for start_all_awgs
        """
        self.__controller.set_awg_start_stop("ALL", "START")
        print("All AWGs started.")

    # ------------------------------------------------------------
    def stop_all_awgs(self) -> None:
        """
        Docstring for stop_all_awgs
        """
        self.__controller.set_awg_start_stop("ALL", "STOP")
        print("All AWGs stopped.")

    #-------------------------------------------------

    def sync_all_channels(self) -> None:
        """
        Trigger synchronous update of all DAC channels on both boards.
        
        Both boards must be in synchronous mode first.
        """

        self.__controller.update_board_channels("LH")
    # ------------------------------------------------------------
    @staticmethod
    def cmd_voltage(channel: int, voltage: float) -> str:
        """Build a SET voltage command string for use with multi_set()."""
        return f"{channel} {BaspiLnhrdac2aController.vval_to_dacval(voltage):x}"
    
    # ------------------------------------------------------------
    @staticmethod
    def cmd_status(channel: int, status: str) -> str:
        """Build a SET status command string for use with multi_set()."""
        return f"{channel} {status}"
    
    # ------------------------------------------------------------
    @staticmethod
    def cmd_bandwidth(channel: int, bandwidth: str) -> str:
        """Build a SET bandwidth command string for use with multi_set()."""
        return f"{channel} {bandwidth}"
    
    # ------------------------------------------------------------
    def multi_set(self, commands: list[str]) -> list[str]:
        """
        Send multiple SET commands in a single round-trip.
        Up to 1000 commands can be combined. DAC, status and bandwidth
        commands can be freely mixed. WAV and POLY commands must not be 
        mixed with other command types.

        Use the cmd_voltage(), cmd_status(), cmd_bandwidth() helpers
        to build command strings without manual hex conversion.

        Parameters:
        commands: list of SET command strings as per the programmers manual

        Returns:
        list: DAC-Error Codes, one per command. All "0" on success.

        Example:
        DAC.multi_set([
            DAC.cmd_status(1, "ON"),
            DAC.cmd_voltage(1, 1.0),
            DAC.cmd_bandwidth(1, "HBW"),
        ])
        """
        result = self.__controller.write_multi(commands)
        print(f"multi_set: {len(result)} command(s) sent, all OK")
        return result

    # ------------------------------------------------------------
    def set_channels_dacvalue_multi(self, channel_voltage_map: dict) -> list[str]:
        """
        Set multiple DAC channels to specific voltages in one round-trip.

        Parameters:
        channel_voltage_map: dict mapping channel number (int) to voltage (float)

        Returns:
        list: DAC-Error Codes, one per channel. All "0" on success.
        """
        result = self.__controller.set_channels_dacvalue_multi(channel_voltage_map)
        for ch, v in channel_voltage_map.items():
            print(f"  ch{ch}: {v:+.6f} V")
        return result

    # ------------------------------------------------------------
    def set_channels_status_multi(self, channel_status_map: dict) -> list[str]:
        """
        Set multiple DAC channels on or off in one round-trip.

        Parameters:
        channel_status_map: dict mapping channel number (int) to status ("ON" or "OFF")

        Returns:
        list: DAC-Error Codes, one per channel. All "0" on success.
        """
        result = self.__controller.set_channels_status_multi(channel_status_map)
        for ch, status in channel_status_map.items():
            print(f"  ch{ch}: {status}")
        return result

    # ------------------------------------------------------------
    def set_channels_bandwidth_multi(self, channel_bandwidth_map: dict) -> list[str]:
        """
        Set the bandwidth of multiple DAC channels in one round-trip.

        Parameters:
        channel_bandwidth_map: dict mapping channel number (int) to bandwidth ("LBW" or "HBW")

        Returns:
        list: DAC-Error Codes, one per channel. All "0" on success.
        """
        result = self.__controller.set_channels_bandwidth_multi(channel_bandwidth_map)
        for ch, bw in channel_bandwidth_map.items():
            print(f"  ch{ch}: {bw}")
        return result

    # ------------------------------------------------------------
    def reconnect(self, attempts: int = 10, wait_between_attempts: float = 5.0):
        import pyvisa
        address = getattr(self, "_address", None) or getattr(self, "address", None)
        visalib = getattr(self, "visalib", None)
        if address is None:
            raise RuntimeError("reconnect(): no VISA address available.")

        last_exc = None
        for attempt in range(1, attempts + 1):
            old = getattr(self, "visa_handle", None)
            if old is not None:
                try: old.close()
                except Exception: pass
            old_rm = getattr(self, "resource_manager", None)
            if old_rm is not None:
                try: old_rm.close()          
                except Exception: pass
            self.visa_handle = None

            try:
                rm = pyvisa.ResourceManager(visalib or "")
                handle = rm.open_resource(address)
                handle.write_termination = "\r\n"
                handle.read_termination = "\r\n"
                self.visa_handle = handle
                self.resource_manager = rm
                idn = self.get_idn()
                print(f"[reconnect] OK: {idn.get('model')} S/N {idn.get('serial')}")
                return self
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    print(f"[reconnect] attempt {attempt}/{attempts} failed: {exc!r}\n"
                        f"Make sure 'Restart Telnet now!' was done on the device; "
                        f"retrying in {wait_between_attempts:.1f}s...")
                    sleep(wait_between_attempts)
        raise RuntimeError(
            "reconnect(): all attempts failed. Most common cause: the device-side "
            "Telnet socket was not freed. On the instrument: 'Restart the device' "
            "-> 'Restart Telnet now!', then retry."
        ) from last_exc


# main -----------------------------------------------------------------

if __name__ == "__main__":

    # a little example on how to use this driver

    station = Station()
    dac = BaspiLnhrdac2a('LNHRDAC', 'TCPIP0::192.168.0.5::23::SOCKET')
    station.add_component(dac)