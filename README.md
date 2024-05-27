# Mouse-V1

Code to produce results for IIB Project at the University of Cambridge.

1. The weights generation and network execution classes are in the `rat` folder.
2. As most files will import from `rat`, files which are in folders will likely not work unless `rat` is installed into an env or `rat` is moved into the folder.
3. All plot scripts are in the `plot_scripts` folder.
4. All scripts for the CNN and transformers are in the `deeplearning` folder
5. `legacy` contains all old scripts. Most of these code contains experiments which did not make it into the final report due to bad results. As these code are old, they are unlikely to be compatible with the current version of `rat` and should only be used as reference.
6. `utils` contains useful functions which are repetitive across experiments and scripts. In reality, this should have been included into `rat`.
7. `parameter_tester` test certain parameter conditions as well as format the parameter values to work which each script
8. `rodents_plotter` will plot generic graphs and plots for quick analysis of parameter values.

If you have any questions regarding this repo, please feel free to contact me.
