"""Layer-agnostic simulation core: signals, blocks, graphs and run drivers.

Nothing in this package is specific to the physical layer, so the same machinery
can later host MAC / network-layer blocks.
"""

from linkstudio.core.block import Block, BlockError, BlockParams, Port
from linkstudio.core.composite import Subsystem
from linkstudio.core.context import RunContext
from linkstudio.core.graph import Endpoint, Graph, GraphError
from linkstudio.core.registry import available_blocks, block_catalogue, create_block, register_block
from linkstudio.core.scenario import Scenario, graph_from_dict, graph_to_dict, load_scenario, save_scenario
from linkstudio.core.signal import Kind, Signal
from linkstudio.core.simulation import Simulation, SimulationResult, StopCriteria
from linkstudio.core.sweep import Sweep, SweepResult

__all__ = [
    "Block", "BlockError", "BlockParams", "Endpoint", "Graph", "GraphError", "Kind", "Port", "RunContext",
    "Scenario", "Signal", "Simulation", "SimulationResult", "StopCriteria", "Subsystem", "Sweep",
    "SweepResult", "available_blocks", "block_catalogue", "create_block", "graph_from_dict",
    "graph_to_dict", "load_scenario", "register_block", "save_scenario",
]
