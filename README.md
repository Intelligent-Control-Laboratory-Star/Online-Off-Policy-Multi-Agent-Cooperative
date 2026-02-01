# Online-Off-Policy-Multi-Agent-Cooperative
Handling Stability Control for FWID-EVs Under Tyre Blowout: An Online Off-Policy Multi-Agent Cooperative Approach

Project Overview
This project integrates CarSim 2020 with MATLAB/Simulink R2021b to simulate and control vehicle behavior under multiple tire blowout scenarios.

There are four blowout environments, which are switched via CarSim Events. The control algorithm is implemented in Simulink and the controller logic is switched using a Simulink S-Function.

Environment Setup
Software Requirements
CarSim 2020
MATLAB R2021b (Simulink required)
Microsoft Visual Studio / supported compiler for S-Function build (use the compiler supported by MATLAB R2021)

Installation & Configuration
1) Install CarSim 2020
Install CarSim 2020 normally.
Verify that the CarSim Simulink interface is installed (CarSim provides Simulink integration files/templates).
2) Install MATLAB/Simulink R2021b
Install MATLAB R2021b and Simulink.
Configure a supported C/C++ compiler in MATLAB (required for S-Functions):

In MATLAB Command Window, run:
mex -setup
Select a supported compiler.
3) Connect CarSim with Simulink
Open the CarSim dataset for this project.
Ensure the simulation runs in Simulink co-simulation mode (CarSim provides a Simulink model or export function).
Open the corresponding Simulink model (.slx) and confirm CarSim blocks (or CarSim S-Function block) are correctly linked.
Blowout Environment Switching (CarSim Events)
Four Blowout Environments
This project contains four different tire blowout environments.


They are implemented and switched by CarSim “Events” configuration.

How Switching Works
In CarSim, each blowout case is defined as an Event (e.g., timing, wheel position, pressure drop profile, etc.).
The active scenario is selected by enabling the corresponding Event (or switching the Event set, depending on the dataset design).
Note: The exact event names and parameters depend on the CarSim dataset provided with this project.

Control Algorithm Switching (Simulink S-Function)
Controller Architecture
The control algorithm runs in Simulink.
Controller mode/strategy is switched using a custom S-Function.
How Switching Works
The S-Function encapsulates the controller logic and provides a switching interface (e.g., via an input signal, parameter, or enum-like mode index).
During simulation, different controller behaviors can be activated by changing the switch signal/parameter.
Typical Run Procedure
Open the CarSim dataset and select the desired Event (blowout environment).
Open the Simulink model.
Set the controller switch (S-Function mode selection) as needed.
Run the simulation from Simulink (or from CarSim depending on your workflow).
Notes / Troubleshooting
If the S-Function fails to compile:
Re-run mex -setup
Confirm the compiler version matches MATLAB R2021 support
Clean and rebuild the S-Function
If CarSim blocks are missing in Simulink:
Ensure CarSim 2020 Simulink integration is installed
Verify CarSim paths are correctly added (as required by CarSim)
