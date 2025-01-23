To edit any of the values the program is supposed to calculate for, please refer to the 'input.json' file.
There you can input the data according to the task at hand.
Below is the explanation of each variabe

n - number of elements in x direction
m - number of elements in y direction
len_x - length of the material
len_y - width of the material

When using Dirichlet boundary conditions, temperatures can be determined.
(if you do not specify Dirichlet BC for a given side, but still input a temperature value the program will ignore it;
for clarity one can input 'null' to the variable not in use)
t_left - temperature at the left boundary [K]
t_right - temperature at the right boundary [K]
t_top - temperature at the top boundary [K]
t_bottom - temperature at the bottom boundary [K]

When using Neumann boundary conditions, the heat flux can be determined.
(if you do not specify Neumann BC for a given side, but still input a heat flux value the program will ignore it;
for clarity one can input 'null' to the variable not in use)
q_left - heat flux at theleft boundary [W/m^2]
q_right - heat flux at the right boundary [W/m^2]
q_top - heat flux at the top boundary [W/m^2]
q_bottom - heat flux at the bottom boundary [W/m^2]

To specify boundary conditions at a given side input one of the options ("Neumann" or "Dirichlet") to an appropriate variable.
bc_bottom - boundary conditions at the bottom side
bc_left - boundary conditions at the left side
bc_right - boundary conditions at the right side
bc_top - boundary conditions at the top side

If you wish to specify two different materials within the body it is possible by changing the following variables.
If you wish to calculate for only one material set 'x' to the length of the material or 'k2' to 'null'
x - length of the first layer
k1 - thermal conductivity coefficient for the first material [W/mK]
k2 - thermal conductivity coefficient for the second material [W/mK]

If you wish to add a point heat source you can do it by changing the following variables.
node - the node at which you wish to add the point heat source
heat_value - the temperature of the point source [K]
If you do not wish to add a point source, you can either set a node out of range or set the node to 'null'.

When plotting the results for temperature along the line, one can determine whether the plot 
should be of temperatures depending on x ("horizontal) or y ("vertical") by inputing that to the 'xy_plot' variable

Not following the instructions might results in errors while running the program.
When inputting 'null' into a variable, do it without the quotation marks.
