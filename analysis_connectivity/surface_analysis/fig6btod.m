addpath /Users/ulises/Documents/MATLAB/gifti
recording_times = [533];
num_recording_times = length(recording_times);
rates_file = zeros(num_recording_times,89, 2);
num_areas = size(rates_file,2);
% load in left hemisphere inflated surface
l_inflated = gifti('surface_files/MacaqueYerkes19.L.inflated.32k_fs_LR.surf.gii');
% load in LH kennedy atlas (91 regions)
kennedy_atlas_91 = gifti('surface_files/kennedy_atlas_91.label.gii');


%%
% load in a gifti file of the right type in order to get a
% template to write over
example = gifti('surface_files/cortical_thickness.func.gii');
num_vertices = length(example.cdata);
example.cdata = zeros(num_vertices,num_recording_times);
%%
% get area List in Donahue order
% areaList_Donahue = kennedy_atlas_91.labels.name(2:end)';
% Note 18-Jan-2021 - for some reason the labels.name method has stopped
% correctly reading the label file.
load surface_files/areaList_Donahue.mat
load surface_files/names_ctx.mat
load surface_files/Fig6/areas_surface_visual.mat
load surface_files/Fig6/areas_surface_salience.mat
load surface_files/Fig6/areas_surface_auditory.mat

[~, ulises_areas_in_Donahue_idx] = ismember(names_ctx,areaList_Donahue);

rate_map = example;





current_time = 1;
for current_parcel = 1:num_areas
    vertices_in_parcel = find(kennedy_atlas_91.cdata==ulises_areas_in_Donahue_idx(current_parcel)); % note kennedy_atlas_91.cdata ranges from 0-91, not 1-92
   rate_map.cdata(vertices_in_parcel,current_time) = areas_surface_auditory(1,current_parcel);
end



fig = plot(l_inflated, rate_map);
%colormap([1.0 1.0 1.0;0.467, 0.173, 0.494]); %salience
%colormap([1.0 1.0 1.0;0.922, 0.643, 0.165]); %visual
colormap([1.0 1.0 1.0;0.224, 0.510, 0.259]); %auditory
clim([0, 0.45])
material([0.3, 0.6, 0])
%colormapeditor