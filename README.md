# LISA-Sensitivity-Response

<mark style="background: yellow" > **如果您无法在 GitHub 网页端通过浏览器直接查看 .png、.mp4、.pdf 等文件，建议将仓库克隆到本地进行访问。由此给您带来的不便，我深表歉意!**  </mark>

## Overview
This repository contains my submission for the ICTP-AP 2025 selection task: "LISA Detector Response Sensitivity."

## Repository Structure
### code/:
 - 'main.py': This program defines the functions that will be used later and plots the analytical expressions of <(4F^{+/x}_{X})^2> and <F^{+}_{X} (F^{x}_{X})^{*} + (F^{+}_{X})^{*} F^{x}_{X}>
- 'numerical_and_approximation.py': This program computes the analytical expression and approximation of <(4F^{+}_{X})^2> and plots 
- 'responsefig.py': This program plots the fig of average GW response for TDI-X2.0 (Numerical and Semi-analytical).
 - 'sensitivity.py': This program computes sensitivity (SciRD) and plots.
 
### docs/: LaTeX source and compiled PDF.
 - 'Calculation_of_the_Average_GW_Response_for_TDI-X2.0.tex'
 - 'Calculation_of_the_Average_GW_Response_for_TDI-X2.0.pdf'
 
### results/: Output files.
 - 'mainfig.png': averaged antenna response functions and cross term.
 - 'Numerical_approximation.png': averaged antenna response and correction term vs frequency.
 - 'numerical&semi_response.png': average GW response for TDI-X2.0 (numerical and semi-analytical).
 - 'sensitivity.png': sensitivity curve for a single TDI-X observable using SCiRD

## Environment
 Operating System: Ubuntu 24.04.2 LTS
