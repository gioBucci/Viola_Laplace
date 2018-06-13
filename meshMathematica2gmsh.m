% Converts Giovanna's output files from Mathematica into a gmsh
% mesh. Gio's files are an array of coordinates and a triangle 
% connectivity table.
path='./mesh/';
conn=dlmread(strcat(path,'elementsPolycrystal_2Drectangle.csv'));
coords=dlmread(strcat(path,'nodesPolycrystal_2Drectangle.csv'));

materialID=dlmread(strcat(path,'GrainID.csv'));

outfilename=strcat(path,'polycrystal_2Drectangle.msh');

nodes=length(coords);
elements=length(conn);
edges=0; %length(rightEdges);

scale=1;
% 2D
gmsh_coords=[(1:nodes)', coords*scale, zeros(nodes,1)];
% 3D
% gmsh_coords=[(1:nodes)', coords*scale];

% set up gmsh connectivity (without material ID)
% gmsh_conn=[(1:elements)', repmat([3 1 0], elements, 1), conn];

% set up gmsh connectivity (with material ID) according to file format 1
gmsh_conn=[(1:elements)', repmat(3, elements, 1), materialID, repmat([0 4], elements, 1), conn];
% the last entry, before the connectivity, is the number of nodes per element
% 2D->4 3D->8

% gmsh_conn=[(1:elements)', repmat([3 1], elements, 1), materialID, conn];
% FORMAT of element data for gmsh file format 2 (according to deal II
% interpretation
% element number 
% element type (3 -> quadrilateral, 5-> hexahedra)
% numeber of tags (set to 1)
% tag1 (material ID)

% set up the boundary indicators by defining lines along the internal
% boundaries (electrode-electrolyte interface)
% gmsh_leftbound=[(elements+1:elements+edges)', repmat([1 104 104 2], edges, 1), leftEdges];
% gmsh_rightbound=[(elements+edges+1:elements+2*edges)', repmat([1 105 105 2], edges, 1), rightEdges];
% FORMAT of line data
% line number
% element type (1 -> line)
% boundary ID
% physical line
% dummy

file=fopen(outfilename,'w');

% fprintf(file,'$MeshFormat\n');
% fprintf(file,'2.2 0 8\n');
% fprintf(file,'$EndMeshFormat\n');

% fprintf(file,'$Nodes\n');
fprintf(file,'$NOD\n');
fprintf(file,'%u\n',nodes);

fclose(file);

file=fopen(outfilename,'a');

dlmwrite(outfilename, gmsh_coords, '-append', 'delimiter', ' ', 'precision', '%g');

% fprintf(file,'$EndNodes\n');
fprintf(file,'$ENDNOD\n');

% fprintf(file,'$Elements\n');
fprintf(file,'$ELM\n');
fprintf(file,'%d\n',elements+2*edges);

fclose(file);

dlmwrite(outfilename, gmsh_conn, '-append', 'delimiter', ' ', 'precision', '%u');

% dlmwrite(outfilename, gmsh_leftbound, '-append', 'delimiter', ' ');
% dlmwrite(outfilename, gmsh_rightbound, '-append', 'delimiter', ' ');

file=fopen(outfilename,'a');
% fprintf(file,'$EndElements\n');
fprintf(file,'$ENDELM\n');

fclose(file);

