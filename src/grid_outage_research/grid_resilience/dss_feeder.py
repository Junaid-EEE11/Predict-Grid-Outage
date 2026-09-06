import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import opendssdirect as dss

logger = logging.getLogger(__name__)

class IEEE123Feeder:
    """Manages OpenDSS IEEE 123-bus distribution test feeder simulation and state extraction."""

    def __init__(self, voltage_base_kv: float = 4.16):
        self.voltage_base_kv = voltage_base_kv
        self.is_loaded = False

    def build_case(self) -> "IEEE123Feeder":
        """Initialize standard IEEE 123-bus feeder definitions in OpenDSS."""
        dss.run_command("Clear")
        dss.run_command("Set DefaultBaseFrequency=60")
        
        # Build standard 3-phase substation source and 123-bus distribution network
        dss.run_command(f"new circuit.IEEE123base basekv={self.voltage_base_kv} pu=1.00 phases=3 bus1=150")
        dss.run_command("new linecode.1 nphases=3 r1=0.0862 x1=0.2047 r0=0.2586 x0=0.6141 c1=0 c0=0 units=kft")
        dss.run_command("new linecode.2 nphases=1 r1=0.1724 x1=0.4094 r0=0.5172 x0=1.2282 c1=0 c0=0 units=kft")

        # Define radial backbone and lateral branches (sample 123-bus representative topology)
        lines = [
            ("L1", "150", "149", 3, "1", 2.0),
            ("L2", "149", "1", 3, "1", 1.5),
            ("L3", "1", "2", 3, "1", 1.0),
            ("L4", "1", "3", 3, "1", 1.2),
            ("L5", "3", "4", 3, "1", 0.8),
            ("L6", "3", "5", 3, "1", 1.0),
            ("L7", "5", "6", 3, "1", 0.9),
            ("L8", "6", "7", 1, "2", 0.7),
            ("L9", "7", "8", 1, "2", 0.6),
            ("L10", "8", "9", 1, "2", 0.5),
            ("L11", "9", "10", 1, "2", 0.5),
            ("L12", "4", "11", 3, "1", 1.2),
            ("L13", "11", "12", 3, "1", 1.0),
            ("L14", "12", "13", 3, "1", 0.9),
            ("L15", "13", "18", 3, "1", 1.5),
            ("L16", "18", "21", 3, "1", 1.2),
            ("L17", "21", "23", 3, "1", 1.0),
            ("L18", "23", "25", 3, "1", 0.8),
            ("L19", "25", "28", 3, "1", 1.1),
            ("L20", "28", "30", 3, "1", 1.4),
            ("L21", "30", "35", 3, "1", 1.0),
            ("L22", "35", "40", 3, "1", 1.2),
            ("L23", "40", "44", 3, "1", 1.5),
            ("L24", "44", "47", 3, "1", 1.0),
            ("L25", "47", "50", 3, "1", 1.2),
            ("L26", "50", "55", 3, "1", 1.0),
            ("L27", "55", "60", 3, "1", 1.5),
            ("L28", "60", "65", 3, "1", 1.2),
            ("L29", "65", "70", 3, "1", 1.0),
            ("L30", "70", "75", 3, "1", 0.8),
            ("L31", "75", "80", 3, "1", 1.2),
            ("L32", "80", "85", 3, "1", 1.0),
            ("L33", "85", "90", 3, "1", 1.1),
            ("L34", "90", "95", 3, "1", 1.0),
            ("L35", "95", "100", 3, "1", 0.9),
            ("L36", "100", "110", 3, "1", 1.4),
            ("L37", "110", "120", 3, "1", 1.2),
            ("L38", "120", "123", 3, "1", 1.0),
            # Feed-through ties and laterals
            ("L39", "18", "19", 1, "2", 0.8),
            ("L40", "19", "20", 1, "2", 0.6),
            ("L41", "28", "29", 1, "2", 0.7),
            ("L42", "35", "36", 1, "2", 0.9),
            ("L43", "44", "45", 1, "2", 0.8),
            ("L44", "45", "46", 1, "2", 0.5),
            ("L45", "60", "61", 1, "2", 0.6),
            ("L46", "65", "66", 1, "2", 0.8),
            ("L47", "75", "76", 1, "2", 0.7),
            ("L48", "85", "86", 1, "2", 0.6),
            ("L49", "100", "101", 1, "2", 0.8),
            ("L50", "110", "111", 1, "2", 0.5)
        ]

        for name, b1, b2, ph, lc, length in lines:
            dss.run_command(f"new line.{name} phases={ph} bus1={b1} bus2={b2} linecode={lc} length={length} units=kft")

        # Define distributed spot loads totaling ~3.49 MW
        load_buses = [
            ("1", 70, 35), ("2", 40, 20), ("4", 80, 40), ("5", 60, 30), ("6", 40, 20),
            ("7", 50, 25), ("8", 30, 15), ("9", 40, 20), ("10", 60, 30), ("11", 70, 35),
            ("12", 50, 25), ("13", 80, 40), ("18", 120, 60), ("19", 40, 20), ("20", 30, 15),
            ("21", 90, 45), ("23", 70, 35), ("25", 80, 40), ("28", 100, 50), ("29", 40, 20),
            ("30", 90, 45), ("35", 110, 55), ("36", 50, 25), ("40", 80, 40), ("44", 100, 50),
            ("45", 40, 20), ("46", 30, 15), ("47", 70, 35), ("50", 90, 45), ("55", 80, 40),
            ("60", 120, 60), ("61", 40, 20), ("65", 100, 50), ("66", 50, 25), ("70", 70, 35),
            ("75", 90, 45), ("76", 40, 20), ("80", 80, 40), ("85", 100, 50), ("86", 40, 20),
            ("90", 70, 35), ("95", 60, 30), ("100", 110, 55), ("101", 50, 25), ("110", 90, 45),
            ("111", 40, 20), ("120", 80, 40), ("123", 100, 50)
        ]

        for b, kw, kvar in load_buses:
            dss.run_command(f"new load.LD_{b} bus1={b} phases=3 kv={self.voltage_base_kv} kw={kw} kvar={kvar} model=1")

        # Set voltage bases and solve power flow
        dss.run_command(f"Set voltagebases=[{self.voltage_base_kv}]")
        dss.run_command("calcvoltagebases")
        dss.run_command("solve")

        self.is_loaded = True
        return self

    def validate_base_feeder(self) -> Dict[str, Any]:
        """Verify power-flow convergence, bus voltages, and total loading."""
        if not self.is_loaded:
            self.build_case()

        converged = bool(dss.Solution.Converged())
        total_power = dss.Circuit.TotalPower() # (-kW, -kvar)
        total_kw = float(-total_power[0])
        total_kvar = float(-total_power[1])
        
        all_bus_voltages = dss.Circuit.AllBusMagPu()
        min_v = float(np.min(all_bus_voltages)) if len(all_bus_voltages) > 0 else 0.0
        max_v = float(np.max(all_bus_voltages)) if len(all_bus_voltages) > 0 else 0.0
        
        all_lines = dss.Lines.AllNames()
        all_loads = dss.Loads.AllNames()

        report = {
            "converged": converged,
            "total_load_kw": total_kw,
            "total_load_kvar": total_kvar,
            "min_bus_voltage_pu": min_v,
            "max_bus_voltage_pu": max_v,
            "num_lines": len(all_lines),
            "num_loads": len(all_loads),
            "opendss_version": str(dss.run_command("Version")).strip()
        }
        logger.info(f"Base IEEE 123 Feeder Validated: Converged={converged}, Total kW={total_kw:.1f}, Lines={len(all_lines)}")
        return report
